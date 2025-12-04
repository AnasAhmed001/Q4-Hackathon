---
id: 0004
title: Create book structure setup spec
stage: spec
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20251101
feature: book-structure-setup
branch: 1-book-structure-setup
user: (user provided)
command: /sp.specify Physical AI and Humanoid Robotics Textbook - Book Structure Setup\n\nTarget audience: Students with basic Python and AI/ML knowledge learning embodied intelligence\n\nFocus: Initialize Docusaurus site with high-level structure for 4-module robotics curriculum\n\nSuccess criteria:\n- Docusaurus project initialized and deploying to GitHub Pages\n- Navigation structure for 4 modules + intro/resources sections\n- All major pages created with brief descriptions (200-500 words each)\n- Site builds without errors and navigation works\n\nBook structure:├── Introduction (What is Physical AI, Course Overview, Prerequisites)\n├── Getting Started (Hardware Requirements, Software Setup)\n├── Module 1: ROS 2 (Weeks 3-5) - Overview + placeholders\n├── Module 2: Gazebo & Unity (Weeks 6-7) - Overview + placeholders\n├── Module 3: NVIDIA Isaac (Weeks 8-10) - Overview + placeholders\n├── Module 4: Vision-Language-Action (Weeks 11-12) - Overview + placeholders\n├── Capstone Project (Week 13)\n└── Resources (Hardware Guides, Glossary, References)\nDeliverables:\n1. Configured Docusaurus v3.x project (use context7 for setup docs)\n2. Landing page with course value proposition\n3. Module overview pages (300-500 words each describing focus, tech stack, outcomes)\n4. Hardware requirements page (3 tiers: workstation/edge/robot)\n5. Prerequisites and assessment overview pages\n6. GitHub Actions for automated deployment\n7. Placeholder structure for detailed chapters (filled in iteration 2)\n\nTechnical requirements:\n- Docusaurus v3.x with MDX enabled\n- GitHub Pages deployment\n- Sidebar navigation with collapsible module sections\n- MIT or Apache 2.0 license\n\nNot building yet:\n- Detailed chapter content, code examples, tutorials, or exercises\n- These come in iteration 2 (module-by-module detailed specs)\n\nResearch: Use context7 MCP server for Docusaurus initialization, configuration, and GitHub Pages deployment guidance
labels: ["spec","book-structure-setup"]
links:
  spec: specs/1-book-structure-setup/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-book-structure-setup/spec.md
  - specs/1-book-structure-setup/checklists/requirements.md
tests:

---

## Prompt

Physical AI and Humanoid Robotics Textbook - Book Structure Setup

Target audience: Students with basic Python and AI/ML knowledge learning embodied intelligence

Focus: Initialize Docusaurus site with high-level structure for 4-module robotics curriculum

Success criteria:
- Docusaurus project initialized and deploying to GitHub Pages
- Navigation structure for 4 modules + intro/resources sections
- All major pages created with brief descriptions (200-500 words each)
- Site builds without errors and navigation works

Book structure:├── Introduction (What is Physical AI, Course Overview, Prerequisites)
├── Getting Started (Hardware Requirements, Software Setup)
├── Module 1: ROS 2 (Weeks 3-5) - Overview + placeholders
├── Module 2: Gazebo & Unity (Weeks 6-7) - Overview + placeholders
├── Module 3: NVIDIA Isaac (Weeks 8-10) - Overview + placeholders
├── Module 4: Vision-Language-Action (Weeks 11-12) - Overview + placeholders
├── Capstone Project (Week 13)
└── Resources (Hardware Guides, Glossary, References)
Deliverables:
1. Configured Docusaurus v3.x project (use context7 for setup docs)
2. Landing page with course value proposition
3. Module overview pages (300-500 words each describing focus, tech stack, outcomes)
4. Hardware requirements page (3 tiers: workstation/edge/robot)
5. Prerequisites and assessment overview pages
6. GitHub Actions for automated deployment
7. Placeholder structure for detailed chapters (filled in iteration 2)

Technical requirements:
- Docusaurus v3.x with MDX enabled
- GitHub Pages deployment
- Sidebar navigation with collapsible module sections
- MIT or Apache 2.0 license

Not building yet:
- Detailed chapter content, code examples, tutorials, or exercises
- These come in iteration 2 (module-by-module detailed specs)

Research: Use context7 MCP server for Docusaurus initialization, configuration, and GitHub Pages deployment guidance

## Response snapshot

The specification for "Physical AI and Humanoid Robotics Textbook - Book Structure Setup" is now complete and validated.

**Feature Branch**: `1-book-structure-setup`
**Spec File**: `specs/1-book-structure-setup/spec.md`
**Checklist File**: `specs/1-book-structure-setup/checklists/requirements.md`

All requirements are clear, testable, and measurable. The feature is ready for the planning phase.

## Outcome

- ✅ Impact: Feature branch created, spec directory and spec file initialized, and quality checklist created and validated. User clarification for deployment time incorporated.
- 🧪 Tests: No formal tests run for spec creation, but internal validation against checklist performed.
- 📁 Files:
  - specs/1-book-structure-setup/spec.md created and updated.
  - specs/1-book-structure-setup/checklists/requirements.md created and updated.
- 🔁 Next prompts: The feature is ready for the planning phase (using `/sp.plan`).
- 🧠 Reflection: Successfully created the feature branch, spec directory, spec file, and quality checklist. Handled missing `pwsh` and `create-phr.sh` scripts by manually performing their actions. The spec was updated based on user clarification and passed validation.

## Evaluation notes (flywheel)

- Failure modes observed: `pwsh` command not found, `create-phr.sh` script not found.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Ensure `pwsh` and PHR script are available or provide robust cross-platform alternatives.