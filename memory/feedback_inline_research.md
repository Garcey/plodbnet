---
name: Prefer inline research over Explore agents
description: For codebase questions in this repo, do inline Read/Grep rather than spawning Explore subagents
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
Prefer inline research (Read/Grep tools in the main conversation) over spawning Explore subagents for routine codebase questions in this repo.

**Why:** User interrupted an Explore agent spawn for a 5-point engine-capability question and restated the requirements. The repo is small enough that inline exploration is faster and keeps the context window with the planning work, rather than delegating and re-synthesizing.

**How to apply:** Only spawn an Explore/Plan agent if the user explicitly asks for it. For typical "how does X work in this codebase" questions, use Grep + Read directly in the main conversation.
