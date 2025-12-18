## AI Code Generation Guidelines


**Follow these principles when generating or modifying code**


* **Plan files**: in some cases, instructions may reference a .md file which details a plan for achieving a complex goal, consisting of multiple tasks.
* **One task at a time**: if a plan file includes multiple tasks, DO NOT initiate work on more than one task simultaneously.
* **Progress tracking indicators**: the plan file may include progress indicators for the tasks:
 * [STATUS: PLANNED] - indicates that this task is not currently active.
 * [STATUS: IN PROGRESS] - indicates that there are currently AI sessions aimed at performing this task
 * [STATUS: PENDING VERIFICATION] - indicates that the AI has completed the work, but it was not yet verified by a human
 * [STATUS: NEEDS MORE WORK] - indicates that the human verification revealed the need for changes.
 * [STATUS: COMPLETE] - indicates that a human has confirmed that the task was completed.
* **Ask for permission**: do not proceed with the implementation of any task before getting permission to do so.
* **Update the plan**: if a plan file is used, and it includes progress indicators, you should update the relevant progress indicators as you initiate
work on tasks and make progress with them.  NEVER mark a task as "COMPLETE" without explict human approval, even if you have performed a verification.