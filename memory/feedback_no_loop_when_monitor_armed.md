---
name: Don't run /loop fallback when a Monitor is already streaming events
description: If a persistent Monitor is reliably emitting the events you'd report on, skip the /loop ScheduleWakeup chain — the fallback ticks are pure overhead
type: feedback
originSessionId: c75d2dd1-9cd3-487d-a246-81e40fa7c832
---
When a persistent Monitor is armed and reliably delivering the events I'd otherwise check for, do NOT run a /loop ScheduleWakeup chain in parallel. Stop the loop and let the Monitor be the sole wake signal.

**Why:** The user explicitly called out: "No point in having you run a loop to check the log when the monitor is already doing it." Every /loop fallback tick spends an SSH round-trip + a wakeup token to query state the Monitor already streams. With a working Monitor the fallback ticks add no information and waste budget.

**How to apply:**
- If user arms (or asks me to arm) a Monitor that reliably catches the events I'd report on, do not also chain ScheduleWakeup on a dynamic /loop.
- Continue handling Monitor task-notification events as they arrive — just don't re-arm a fallback wakeup after each one.
- The /loop dynamic-mode skill says "the Monitor is primary wake signal, ScheduleWakeup is fallback." Treat fallback as load-bearing only if Monitor coverage is doubtful (poll gaps, SSH instability, ambiguous filter). When the Monitor has been emitting cleanly for many events in a row, it isn't doubtful.
- If user explicitly says "stop the loop, keep the monitor," only omit the ScheduleWakeup — do NOT TaskStop the Monitor.
