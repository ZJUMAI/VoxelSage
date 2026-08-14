/**
 * IndexedDB 缓存：将原始 NIfTI 文件 Blob 存储在浏览器中。
 *
 * 只缓存原始二进制（Blob），不缓存解析后的 NiftiVolume。
 * 读取时重新解析原始 Blob → NiftiVolume，避免 DataCloneError。
 *
 * 存储结构（每个上传的 NIfTI 文件一条记录）：
 *   {
 *     caseId: string,   // key — 格式 'nifti:' + fileId
 *     fileId?: string,  // 后端返回的 file_id
 *     fileName: string,
 *     blob: Blob,       // 原始文件 Blob（可 structured-clone）
 *     size: number,     // 文件大小（字节）
 *     mimeType: string,
 *     createdAt: number,
 *   }
 */

interface CachedNiftiRecord {
  caseId: string;
  fileId?: string;
  fileName: string;
  blob: Blob;
  size: number;
  mimeType: string;
  createdAt: number;
}

const DB_NAME = 'VoxelSageCache';
const STORE_NAME = 'niftiVolumes';
const DB_VERSION = 2;

/** 当前页面 origin，用于跨 origin 诊断 */
const CURRENT_ORIGIN = typeof window !== 'undefined' ? window.location.origin : 'unknown';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        console.log('[niftiCache] 创建 object store:', STORE_NAME);
        db.createObjectStore(STORE_NAME, { keyPath: 'caseId' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => {
      console.warn('[niftiCache] openDB 失败:', req.error, 'origin:', CURRENT_ORIGIN);
      reject(req.error);
    };
    req.onblocked = () => {
      console.warn('[niftiCache] openDB 被阻塞（其他标签页在使用数据库）');
      reject(new Error('数据库被阻塞'));
    };
  });
}

/**
 * 将原始 NIfTI 文件 Blob 保存到 IndexedDB。
 * 只缓存原始二进制，不缓存解析后的 NiftiVolume。
 *
 * 可靠性保证：
 * 1. 等待 tx.oncomplete + req.onsuccess 双确认后才 resolve
 * 2. db.close() 在事务确认后调用
 * 3. 任何错误（openDB 失败、store.put 失败）都会 reject 到调用方
 */
export async function saveNiftiBlobToCache(
  key: string,
  data: {
    fileId?: string;
    fileName: string;
    blob: Blob;
    size: number;
    mimeType: string;
  }
): Promise<void> {
  const START = performance.now();
  console.log(
    '[niftiCache] WRITE_START',
    'key=' + key,
    'fileId=' + (data.fileId || ''),
    'fileName=' + data.fileName,
    'sizeMB=' + (data.size / 1024 / 1024).toFixed(1),
  );

  const db = await openDB().catch(err => {
    console.warn('[niftiCache] WRITE_ERROR openDB 失败:', 'key=' + key, 'err=' + String(err));
    throw new Error('openDB 失败: ' + String(err));
  });

  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  const entry: CachedNiftiRecord = {
    caseId: key,
    fileId: data.fileId,
    fileName: data.fileName,
    blob: data.blob,
    size: data.size,
    mimeType: data.mimeType,
    createdAt: Date.now(),
  };

  return new Promise<void>((resolve, reject) => {
    let putDone = false;
    let txDone = false;

    const finish = () => {
      if (putDone && txDone) {
        const elapsed = (performance.now() - START).toFixed(0);
        console.log('[niftiCache] WRITE_OK', 'key=' + key, 'fileId=' + (data.fileId || ''), 'sizeMB=' + (data.size / 1024 / 1024).toFixed(1), 'elapsed=' + elapsed + 'ms');
        db.close();
        resolve();
      }
    };

    const req = store.put(entry);
    req.onsuccess = () => {
      putDone = true;
      finish();
    };
    req.onerror = () => {
      console.warn('[niftiCache] WRITE_ERROR', 'key=' + key, 'err=' + String(req.error));
      db.close();
      reject(req.error || new Error('store.put 未知错误'));
    };

    tx.oncomplete = () => {
      txDone = true;
      finish();
    };
    tx.onerror = () => {
      if (!putDone) {
        console.warn('[niftiCache] WRITE_ERROR 事务失败', 'key=' + key, 'err=' + String(tx.error));
        db.close();
        reject(tx.error || new Error('事务未知错误'));
      }
    };
  });
}

/**
 * 从 IndexedDB 加载指定 key 的 NIfTI 原始文件 Blob。
 * 返回 { blob, fileName, fileId?, size } 或 null。
 *
 * 兼容旧记录：没有 blob 字段的旧格式记录视为 READ_MISS。
 */
export async function loadNiftiBlobFromCache(key: string): Promise<{ blob: Blob; fileName: string; fileId?: string; size: number } | null> {
  console.log('[niftiCache] READ_START', 'key=' + key);

  try {
    const db = await openDB().catch(err => {
      console.warn('[niftiCache] READ_ERROR openDB 失败:', 'key=' + key, 'err=' + String(err));
      throw err;
    });

    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const record = await new Promise<CachedNiftiRecord | undefined>((resolve, reject) => {
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => {
        console.warn('[niftiCache] READ_ERROR store.get 失败:', 'key=' + key, 'err=' + String(req.error));
        reject(req.error);
      };
    });

    db.close();

    if (!record) {
      console.log('[niftiCache] READ_MISS', 'key=' + key);
      return null;
    }

    // 兼容旧记录：没有 blob 字段的旧格式视为无效
    if (!record.blob || !(record.blob instanceof Blob) || record.blob.size === 0) {
      console.log('[niftiCache] READ_MISS', 'key=' + key, 'reason=旧格式记录或无有效 blob');
      return null;
    }

    console.log('[niftiCache] READ_HIT', 'key=' + key, 'fileId=' + (record.fileId || ''), 'fileName=' + record.fileName, 'sizeMB=' + (record.size / 1024 / 1024).toFixed(1));

    return {
      blob: record.blob,
      fileName: record.fileName,
      fileId: record.fileId,
      size: record.size,
    };
  } catch (err) {
    console.warn('[niftiCache] READ_ERROR 读取失败:', 'key=' + key, 'err=' + String(err));
    return null;
  }
}

/**
 * 从 IndexedDB 删除指定 case 的 NIfTI 体数据。
 */
export async function deleteNiftiVolumeFromCache(caseId: string): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    await new Promise<void>((resolve, reject) => {
      const req = store.delete(caseId);
      req.onsuccess = () => { db.close(); resolve(); };
      req.onerror = () => { db.close(); reject(req.error); };
    });
  } catch (err) {
    console.warn('[niftiCache] 删除缓存失败:', err);
  }
}

/**
 * 列出所有已缓存 NIfTI 的 case ID。
 */
export async function getAllCachedCaseIds(): Promise<string[]> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    return new Promise((resolve, reject) => {
      const req = store.getAllKeys();
      req.onsuccess = () => {
        db.close();
        resolve(req.result as string[]);
      };
      req.onerror = () => {
        console.warn('[niftiCache] getAllKeys 失败:', req.error);
        db.close();
        reject(req.error);
      };
    });
  } catch (e) {
    console.warn('[niftiCache] getAllCachedCaseIds 异常:', e);
    return [];
  }
}

/**
 * 清除全部 NIfTI 缓存（版本变更等场景）。
 */
export async function clearAllNiftiCache(): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    return new Promise((resolve, reject) => {
      const req = store.clear();
      req.onsuccess = () => { db.close(); resolve(); };
      req.onerror = () => { db.close(); reject(req.error); };
    });
  } catch {
    // 静默失败
  }
}

/**
 * 诊断辅助：在浏览器控制台调用 window.__niftiDiagnostic() 查看缓存状态。
 */
if (typeof window !== 'undefined') {
  (window as any).__niftiDiagnostic = async function() {
    console.log('═══════ NIfTI 缓存诊断 ═══════');
    console.log('[诊断] origin:', CURRENT_ORIGIN);
    try {
      const ids = await getAllCachedCaseIds();
      console.log('[IndexedDB] 已缓存 ' + ids.length + ' 条记录:', ids);
      if (ids.length > 0) {
        for (const id of ids) {
          const record = await loadNiftiBlobFromCache(id);
          if (record) {
            console.log('[IndexedDB] 有效记录:', 'key=' + id, 'fileName=' + record.fileName, 'sizeMB=' + (record.size / 1024 / 1024).toFixed(1));
          } else {
            console.log('[IndexedDB] 旧格式记录(跳过):', 'key=' + id);
          }
        }
      }
    } catch (e) {
      console.warn('[IndexedDB] 读取失败:', e);
    }
    try {
      const selectedCase = localStorage.getItem('CLINICAL_SELECTED_CASE_ID');
      const caseList = localStorage.getItem('CLINICAL_PATIENT_CASES');
      const volumes = localStorage.getItem('CLINICAL_UPLOADED_VOLUMES');
      console.log('[localStorage] selected case:', selectedCase);
      console.log('[localStorage] cases:', caseList ? JSON.parse(caseList).map((c: any) => c.id) : 'none');
      console.log('[localStorage] uploaded volumes:', volumes ? JSON.parse(volumes).map((v: any) => v.file_id) : 'none');
    } catch (e) {
      console.warn('[localStorage] 读取失败:', e);
    }
    console.log('════════════════════════════════');
  };
}
