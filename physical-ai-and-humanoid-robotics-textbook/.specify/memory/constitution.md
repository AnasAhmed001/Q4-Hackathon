<!--
Version change:  → v1.0.0
List of modified principles: All new
Added sections: Key Standards, Content Structure, Technical Constraints, Success Criteria, Governance
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- .specify/templates/commands/*.md: ⚠ pending
Follow-up TODOs: None
-->
# Physical AI and Humanoid Robotics Textbook Constitution
<!-- Example: Spec Constitution, TaskFlow Constitution, etc. -->

## Core Principles

### Pedagogical clarity
Content must be accessible to undergraduate/graduate students from **Computer Science, Mechanical Engineering, and Electrical Engineering backgrounds** with **intermediate Python proficiency (OOP, error handling, modules)** and **basic robotics concepts (robot types, sensors, actuators)**. Content aims for a Flesch-Kincaid grade level of 10-12, and uses active voice in at least 75% of sentences.

### Course Focus
This textbook is intended as a **survey course**, emphasizing breadth across various tools and concepts rather than deep-dive mastery of a few. Students are expected to gain a foundational understanding and practical exposure to a wide range of topics in Physical AI and Humanoid Robotics.

### Physical Hardware Usage
Some components of the assessment and learning experience MUST involve **physical hardware**, reinforcing practical application alongside simulation.

### Pedagogical clarity

### Technical accuracy
All technical claims, formulas, and implementations must be verified against authoritative sources (research papers, official documentation, industry standards)

### Hands-on learning
Include practical examples, code snippets, and real-world applications for every major concept

### Progressive complexity
Structure content from foundational concepts to advanced topics with clear learning pathways. Each chapter must explicitly state prerequisites, ensuring a logical progression of knowledge.

### Open educational resource (OER)
Content must be freely accessible, reusable, and maintainable by the community

## Key Standards

- All code examples must be tested and functional
- Technical diagrams and visualizations required for complex concepts
- Each chapter must include: learning objectives, key concepts, examples, exercises, and further reading
- Source citation: IEEE format for academic papers, inline links for documentation
- Code standards: Python 3.10+ with type hints, well-commented, following PEP 8
- Accessibility: WCAG 2.1 AA compliance for deployed site
- Plagiarism: All content must pass automated plagiarism checks (e.g., Turnitin, Copyscape) with an originality score of at least 90% (excluding correctly cited references)
- Version control: All changes tracked with meaningful commit messages

## Content Review Process

- All new or substantially revised content must undergo a formal review process.
- **Technical Review:** Conducted by subject matter experts to verify accuracy, completeness, and alignment with current research and industry practices.
- **Pedagogical Review:** Conducted by educators or instructional designers to ensure clarity, accessibility, progressive complexity, and adherence to learning objectives.
- **Accessibility Review:** Content must be checked for WCAG 2.1 AA compliance by an accessibility specialist.
- **Code Review:** All code examples must be reviewed for functionality, adherence to code standards, and security best practices.
- Review feedback must be addressed and approved before publication.

## Content Structure

- Docusaurus-based organization with clear navigation hierarchy
- Markdown files with MDX support for interactive components
- Modular chapters that can be read independently or sequentially
- Glossary of technical terms maintained across all chapters
- Bibliography with categorized resources (foundational papers, tools, frameworks, datasets)

## Technical Constraints

- Platform: Docusaurus v3.x
- Deployment: GitHub Pages with automated CI/CD
- Repository: Public GitHub repository with Apache 2.0 or MIT license
- Assets: Images optimized (<500KB), videos hosted externally
- Build time: <5 minutes for full site generation
- Browser support: Modern browsers (Chrome, Firefox, Safari, Edge - last 2 versions)

## Success Criteria

- Content covers Physical AI fundamentals through advanced humanoid robotics applications
- All code examples execute successfully
- Site loads within 3 seconds on standard broadband
- Zero broken links or missing references
- Passes accessibility validation
- Community contributions enabled through clear CONTRIBUTING.md guidelines
- Suitable for use in academic courses or self-study programs

## Governance
<!-- Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->
This constitution supersedes all other project practices. Amendments require a documented proposal, team approval, and a clear migration plan for any affected components. All pull requests and reviews must verify compliance with these principles.

**Version**: v1.0.3 | **Ratified**: 2025-12-04 | **Last Amended**: 2025-12-04
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->