"""GeoSurge Port B Skills system.

Skills are modular, executable analysis units that Port A (LLM) can
autonomously invoke via POST /api/skills/run.

Built-in skills live under skills/builtin/<name>/ and use the
same run(ctx) interface as user-uploaded skills.
"""
from .engine import SkillEngine
from .models import SkillContext, SkillManifest

__all__ = ["SkillEngine", "SkillContext", "SkillManifest"]
