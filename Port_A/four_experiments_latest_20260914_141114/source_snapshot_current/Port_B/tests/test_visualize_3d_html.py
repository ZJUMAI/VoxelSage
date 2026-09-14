import sys
import types
import unittest


def _unused_dependency(name):
    module = types.ModuleType(name)

    def unavailable(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be used while generating HTML")

    return module, unavailable


nibabel, _nibabel_unavailable = _unused_dependency("nibabel")
numpy, _numpy_unavailable = _unused_dependency("numpy")
numpy.ndarray = object
scipy, _scipy_unavailable = _unused_dependency("scipy")
scipy_ndimage, _scipy_ndimage_unavailable = _unused_dependency("scipy.ndimage")
scipy_ndimage.zoom = _scipy_ndimage_unavailable
scipy_ndimage.gaussian_filter = _scipy_ndimage_unavailable
skimage, _skimage_unavailable = _unused_dependency("skimage")
skimage_measure, _skimage_measure_unavailable = _unused_dependency("skimage.measure")
skimage_measure.marching_cubes = _skimage_measure_unavailable
skimage_measure.mesh_surface_area = _skimage_measure_unavailable

sys.modules.setdefault("nibabel", nibabel)
sys.modules.setdefault("numpy", numpy)
sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.ndimage", scipy_ndimage)
sys.modules.setdefault("skimage", skimage)
sys.modules.setdefault("skimage.measure", skimage_measure)

from Visualization.visualize_3d import _make_threejs_html


class OrganTreeResizeHtmlTests(unittest.TestCase):
    def test_organ_tree_has_horizontal_resize_handle_and_bounds(self):
        html = _make_threejs_html()

        self.assertIn('id="organ-tree-resize-handle"', html)
        self.assertIn("const ORGAN_TREE_MIN_WIDTH = 190;", html)
        self.assertIn("const ORGAN_TREE_MAX_WIDTH = 480;", html)
        self.assertIn(
            "const minimum = Math.min(ORGAN_TREE_MIN_WIDTH, viewportMaximum);",
            html,
        )
        self.assertIn("organTree.style.width = `${nextWidth}px`;", html)

    def test_organ_tree_resize_logic_does_not_set_height(self):
        html = _make_threejs_html()
        start = html.index("const ORGAN_TREE_MIN_WIDTH = 190;")
        end = html.index("// Organ tree resize end")
        resize_logic = html[start:end]

        self.assertNotIn(".style.height", resize_logic)
        self.assertIn("pointerdown", resize_logic)
        self.assertIn("pointermove", resize_logic)
        self.assertIn("setPointerCapture", resize_logic)

    def test_viewer_uses_shared_clinical_workspace_styles(self):
        html = _make_threejs_html()

        self.assertIn("--ui-primary: #0f6f78;", html)
        self.assertIn("--left-panel-width: 232px;", html)
        self.assertIn("--right-panel-width: 276px;", html)
        self.assertIn("--model-safe-left: 256px;", html)
        self.assertIn("getModelSafeAspect()", html)
        self.assertIn("resizeRendererToView()", html)
        self.assertIn("has-resection-workspace", html)
        self.assertIn('id="right-workspace"', html)
        self.assertIn("top: calc(var(--header-height) + 18px);", html)
        self.assertIn("rightWorkspace.insertBefore(selector, sequencePanel)", html)
        self.assertIn("rightWorkspace.insertBefore(legend, planeSelector || sequencePanel)", html)
        self.assertIn("#color-legend {\n  order: 1;", html)
        self.assertIn("#plane-selector {\n  order: 2;", html)
        self.assertIn("#path-planning-panel {\n  order: 3;", html)
        self.assertIn("panel.dataset.available = 'true';", html)
        self.assertIn("panel.dataset.available = 'false';", html)
        self.assertIn("clearDisplayedSequence(message, revealPanel = true)", html)
        self.assertIn("clearDisplayedSequence('切面已保存，现在可以进行路径规划。', false)", html)
        self.assertIn("pathPanel.dataset.available === 'true'", html)
        self.assertIn("if (pathPanel) pathPanel.style.display = 'none';", html)
        self.assertIn("grid-template-columns: 26px minmax(52px, 1fr) 26px auto auto auto;", html)
        self.assertIn("restoreBtn.addEventListener('click'", html)
        self.assertIn("rebuildEditedPlane(po, po.originalControlPoints", html)
        self.assertIn("fetch('/api/resection-plane/restore'", html)
        self.assertIn("放弃当前切面的所有人工拖动", html)
        self.assertIn("updatePlaneView(activePlane);", html)
        self.assertIn("planeHeading.className = 'panel-heading plane-panel-heading';", html)
        self.assertIn("panel-heading distance-panel-heading", html)
        self.assertIn("distance-legend-body", html)
        self.assertIn("min-height: 30px;", html)
        self.assertIn('id="structure-panel-toggle"', html)
        self.assertIn('id="toggle-path-planning"', html)
        self.assertIn('id="planning-toggles"', html)
        self.assertIn('id="path-planning-toggle" class="planning-toggle"', html)
        self.assertIn("pathPlanningToggle?.addEventListener('change'", html)
        self.assertIn("resection-workspace-open-request", html)
        self.assertIn("setDistanceCollapsed(false);", html)
        self.assertIn("setPlaneCollapsed(false);", html)
        self.assertIn("panel.style.display = pathToggle && pathToggle.checked", html)
        self.assertNotIn("pathToggle.checked = true;", html)
        self.assertIn('id="btn-zoom-in"', html)
        self.assertIn('id="btn-zoom-out"', html)
        self.assertNotIn('id="btn-bottom-reset"', html)
        self.assertIn("path-planning-open-request", html)
        self.assertIn("handlePathPlanningOpenRequest", html)
        self.assertIn("请先保存所选切除面，再规划手术路径。", html)
        self.assertEqual(html.count("fetch('/api/skills/run'"), 2)
        self.assertIn("skill_name: 'plan_resection_sequence'", html)
        self.assertIn("preview_only: true", html)
        self.assertIn('id="distance-panel-toggle"', html)
        self.assertIn('id="plane-panel-toggle"', html)
        self.assertIn("setStructuresCollapsed(true)", html)
        self.assertIn("setDistanceCollapsed", html)
        self.assertIn("setPlaneCollapsed", html)
        self.assertIn("setDistanceCollapsed(true);", html)
        self.assertIn("setPlaneCollapsed(true);", html)
        self.assertIn("const before = legend.getBoundingClientRect();", html)
        self.assertIn("const before = selector.getBoundingClientRect();", html)
        self.assertIn("const wasDragged = legend.dataset.dragX !== undefined", html)
        self.assertIn("const wasDragged = selector.dataset.dragX !== undefined", html)
        self.assertIn("legend.style.translate = `${dragX}px ${dragY}px`;", html)
        self.assertIn("'organ-tree', 'color-legend', 'plane-selector', 'path-planning-panel'", html)
        self.assertIn('class="path-panel-title" data-drag-handle="true"', html)
        self.assertIn("window.matchMedia('(min-width: 821px)')", html)
        self.assertIn("panel.style.translate", html)
        self.assertIn("document.addEventListener('pointermove', movePanel)", html)
        self.assertIn("new MutationObserver(initializeCompactPanels)", html)
        self.assertIn("const before = structurePanel?.getBoundingClientRect();", html)
        self.assertIn("before.left - after.left", html)
        self.assertIn("before.top - after.top", html)
        self.assertIn("body.structures-collapsed", html)
        self.assertIn("body.structures-collapsed #organ-tree {", html)
        self.assertIn("@media (max-width: 820px)", html)
        self.assertIn("@media (max-width: 620px)", html)
        self.assertIn("--right-panel-width: 164px;", html)
        self.assertIn("--model-safe-right: 178px;", html)
        self.assertIn("const safePadding = compactCollapsed ? 0.68", html)
        self.assertIn("transform: translateY(-50%);", html)
        self.assertIn('id="interaction-help"', html)
        self.assertIn('id="btn-help"', html)
        self.assertIn("const setHelpOpen = open =>", html)
        self.assertIn("#interaction-help.open", html)
        self.assertIn("<kbd>右键拖动</kbd><span>平移视图</span>", html)
        self.assertIn("<div id=\"info\">解剖结构与切除规划</div>", html)
        self.assertIn('id="compact-mode-hint"', html)
        self.assertIn("进入全屏模式以使用完整功能", html)
        self.assertIn("@media (max-aspect-ratio: 6/5)", html)
        self.assertIn("#organ-tree,\n  #right-workspace {\n    display: none !important;", html)
        self.assertIn("'btn-reset': '重置视图'", html)
        self.assertIn('class="panel-heading"', html)
        self.assertIn('id="structure-count"', html)
        self.assertIn("const toolbarIcons = {", html)
        self.assertIn('class="button-label"', html)

    def test_toolbar_preserves_icons_when_state_changes(self):
        html = _make_threejs_html()
        rotate_start = html.index("// 7b. Auto-rotate")
        rotate_end = html.index("// 7d. Screenshot")
        toolbar_state_logic = html[rotate_start:rotate_end]

        self.assertNotIn("btnRotate.textContent", toolbar_state_logic)
        self.assertNotIn("btnBg.textContent", toolbar_state_logic)
        self.assertIn("aria-pressed", toolbar_state_logic)

    def test_panel_collapse_does_not_touch_3d_renderer_or_camera(self):
        html = _make_threejs_html()
        self.assertNotIn("structures-panel-layout-change", html)

    def test_cards_use_one_cross_platform_sans_serif_type_system(self):
        html = _make_threejs_html()

        self.assertIn("--ui-font-family: Arial, Helvetica, sans-serif;", html)
        self.assertIn("--ui-card-content-size: 12px;", html)
        self.assertIn("--ui-card-meta-size: 11px;", html)
        self.assertIn("--ui-card-content-size: 11px;", html)
        self.assertIn("--ui-card-meta-size: 10px;", html)
        self.assertIn("font-family: var(--ui-font-family) !important;", html)
        self.assertIn("#organ-tree .tree-item .label,", html)
        self.assertIn("#color-legend .distance-legend-row,", html)
        self.assertIn("#plane-label,", html)
        self.assertIn("#path-planning-panel label {", html)

    def test_cdn_mode_embeds_bezier_surface_as_blob(self):
        # VoxelSage always renders in CDN mode (Three.js library files are
        # intentionally not bundled).  BezierSurface.js is project-authored
        # code, so it must be embedded as a self-contained Blob URL rather
        # than referenced via a relative path that the served HTML cannot
        # resolve from the output directory.
        html = _make_threejs_html(three_sources=None, bezier_source="const X = 1;")

        self.assertIn('id="bs-source"', html)
        self.assertIn("uB = URL.createObjectURL(new Blob([sB]", html)
        self.assertIn('"BezierSurface":uB', html)
        self.assertIn("document.currentScript.after(im)", html)
        # CDN wildcard still resolves OrbitControls/DragControls from unpkg
        self.assertIn("https://unpkg.com/three@0.160.0/examples/jsm/", html)
        # The previously-used relative reference must not leak into CDN mode
        self.assertNotIn('"BezierSurface": "./BezierSurface.js"', html)

    def test_cdn_mode_without_bezier_keeps_static_importmap(self):
        html = _make_threejs_html(three_sources=None, bezier_source=None)

        self.assertIn('<script type="importmap">', html)
        self.assertNotIn('id="bs-source"', html)
        self.assertIn("https://unpkg.com/three@0.160.0/examples/jsm/", html)


if __name__ == "__main__":
    unittest.main()
