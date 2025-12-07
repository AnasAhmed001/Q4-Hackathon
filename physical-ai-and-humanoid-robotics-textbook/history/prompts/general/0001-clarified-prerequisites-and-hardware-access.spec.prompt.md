---
id: 0001
title: Clarified Prerequisites and Hardware Access
stage: spec
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20251101
feature: general
branch: 1-vision-language-action-module
user: user
command: /sp.clarify
labels: ["clarification", "prerequisites", "hardware"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
  - specs/1-ros2-module/spec.md
  - specs/1-digital-twin-module/spec.md
  - specs/1-ai-robot-brain-module/spec.md
  - specs/1-vision-language-action-module/spec.md
tests:
  - N/A
---

## Prompt

My Physical AI & Humanoid Robotics textbook specifications are located at:
- specs/1-ros2-module/spec.md
- specs/1-digital-twin-module/spec.md
- specs/1-ai-robot-brain-module/spec.md
- specs/1-vision-language-action-module/spec.md

Please analyze them for:

1. AMBIGUOUS TERMS
   - What does "Python proficiency" mean as a prerequisite? (beginner vs. advanced? OOP required? async/await knowledge?)
   - What does "basic robotics concepts" mean? (specific math: linear algebra, calculus? kinematics knowledge? control theory?)
   - What constitutes "realistic physics" in Gazebo? (solver accuracy? time-step frequency? collision mesh complexity?)
   - What does "high-fidelity rendering" mean in Unity? (target FPS? polygon count? PBR materials? HDR lighting?)
   - What is "real-time" for VSLAM on Jetson? (30 FPS? 60 FPS? maximum acceptable latency?)
   - What does ">90% word accuracy" mean for voice commands? (WER metric? in what acoustic conditions?)
   - What constitutes a "complex task" in the autonomous humanoid demo? (how many steps? decision points? failure recovery?)

2. MISSING ASSUMPTIONS
   - **Cross-Module Dependencies**: Do students MUST complete modules sequentially, or can they work in parallel? Which concepts from Module 1 are REQUIRED for Module 3?
   - **Hardware Access Model**: Do students own individual hardware kits, or share lab resources? How does this affect deliverable scope?
   - **Development Environment**: Should ALL students use Ubuntu 22.04 native, or is WSL2/Docker acceptable? What about M1/M2 Mac users?
   - **Cloud vs. Local**: Are cloud GPU instances (AWS/Azure) acceptable, or must students have local RTX GPUs? Cost implications?
   - **ROS 2 Distribution**: Which specific ROS 2 version (Humble? Iron? Rolling?)? Will we support multiple versions?
   - **API Costs**: For Module 4, who pays for GPT-4 API usage? Are there rate limits? What's the budget per student?
   - **Dataset Sizes**: For synthetic data generation, what's the minimum/maximum dataset size? Storage requirements per student?
   - **Assessment Format**: Are assessments hands-on labs, written exams, project presentations, or code submissions?
   - **Target Audience Background**: Are students from CS, ME, EE backgrounds? Mixed cohorts? Does curriculum assume specific math prerequisites?

3. INCOMPLETE REQUIREMENTS
   - **Module 1 (ROS 2)**:
     * What specific joint types must the URDF model include? (minimum: revolute + prismatic? or all 6 types?)
     * How many degrees of freedom for the "simplified humanoid"? (7-DOF arm? full 25+ DOF body?)
     * Should launch files support multiple robots, or single-robot scenarios only?
     * What parameter management strategy? (YAML files? launch arguments? dynamic reconfigure?)

   - **Module 2 (Digital Twin)**:
     * What physics solver should be used in Gazebo? (ODE? Bullet? Dart? Trade-offs?)
     * What mesh formats are acceptable for robot models? (STL? DAE? OBJ?)
     * For Unity ML-Agents, are students training policies or just integrating pre-trained models?
     * What constitutes "successful" ROS-Unity bridge communication? (latency thresholds? message types?)
     * Should sensor noise be Gaussian, or more complex (e.g., depth camera ray-drop artifacts)?

   - **Module 3 (AI-Robot Brain)**:
     * What specific VSLAM algorithm should students implement? (ORB-SLAM3? RTAB-Map? Cartographer? Or Isaac's built-in?)
     * For Nav2 bipedal configuration, what custom plugins are REQUIRED vs. OPTIONAL?
     * What does "10,000+ annotated RGB-D frames" mean? (bounding boxes? semantic segmentation? instance masks?)
     * For sim-to-real transfer, what metrics prove "success"? (just localization error? task completion rate?)
     * How much Jetson Orin memory can students use? (affects model quantization strategies)

   - **Module 4 (VLA)**:
     * What's the vocabulary size for voice commands? (20 commands? 100? open-ended?)
     * For LLM task planning, should students fine-tune models or use zero-shot prompting?
     * What vision transformer architecture? (ViT-Base? ViT-Large? DINO? CLIP?)
     * How should multimodal fusion be implemented? (early fusion? late fusion? attention-based?)
     * What does "cognitive architecture" mean practically? (specific frameworks like SOAR, ACT-R? or custom ROS2-based?)

4. SCOPE CONFLICTS
   - **Breadth vs. Depth**: Is this a survey course (touch many tools lightly) or deep-dive course (master fewer tools thoroughly)?
   - **Simulation vs. Physical**: The title says "Physical AI" but most deliverables are simulation-only. What percentage of assessment MUST use physical hardware?
   - **Research vs. Implementation**: Should students read/present research papers, or focus purely on implementation?
   - **Individual vs. Team**: Are all deliverables individual, or are some team-based (especially the capstone)?
   - **Industry vs. Academic Focus**: Should content emphasize production-ready practices (Docker, CI/CD, safety) or research prototyping?

5. SUCCESS CRITERIA CONFLICTS
   - **Module 1**: SC-001 says "15 minutes" to create workspace, but FR-012 requires URDF of "simplified humanoid" - how complex can a humanoid be to model in 15 min?
   - **Module 2**: SC-002 says "60 minutes" to integrate 3 sensors with noise modeling - is this realistic for first-time Gazebo users?
   - **Module 3**: SC-002 requires "<10cm localization error" but doesn't specify environment complexity, sensor quality, or movement speed - under what test conditions?
   - **Module 4**: SC-003 requires ">70% success rate" for end-to-end demo but doesn't define failure modes - does missed grasp count as failure? partial task completion?

6. CROSS-MODULE INTEGRATION
   - Does the Module 4 capstone REQUIRE working implementations from Modules 1-3, or can students use provided scaffolding?
   - Can students substitute tools? (e.g., use MoveIt instead of Nav2? Use Claude instead of GPT-4?)
   - How do assessment timelines interact? Can students reuse code across modules, or must each be standalone?
   - What's the version compatibility matrix? (ROS 2 Humble + Isaac Sim 4.0 + Unity 2022 LTS + Jetson JetPack version?)

7. HARDWARE CONSTRAINTS CLARITY
   - If students CANNOT afford Jetson Orin ($249), what's the fallback? (cloud alternatives? shared lab access? course failure?)
   - For "RTX 4070 Ti minimum" - what happens if a student has RTX 3060 (8GB VRAM)? Can they still complete the course with reduced scene complexity?
   - RealSense D435i is $349 - can students use cheaper alternatives like OAK-D ($149)? or webcam + depth estimation?

List any gaps or questions. Which ones are:
- **CRITICAL (BLOCKING)**: Planning and implementation cannot proceed without clarification.
- **HIGH PRIORITY**: Significantly impacts student success and learning outcomes.
- **MEDIUM PRIORITY**: Affects quality and consistency but workarounds exist.
- **NICE-TO-HAVE**: Improves clarity but doesn't block development.

## Response snapshot

Questions have been asked and answered for Python proficiency, basic robotics concepts, module dependencies, and hardware access model.

- Python Proficiency: Intermediate (OOP, error handling, modules)
- Basic Robotics Concepts: Basic (robot types, sensors, actuators)
- Module Dependencies: Parallel work with some prerequisites
- Hardware Access Model: Individual hardware kits

Updates have been applied to `.specify/memory/constitution.md` and all `specs/*.md` files.

## Outcome

- ✅ Impact: Clarified prerequisites and hardware access model in the constitution and module specifications based on user input, enabling more precise planning and content development.
- 🧪 Tests: N/A
- 📁 Files: .specify/memory/constitution.md, specs/1-ros2-module/spec.md, specs/1-digital-twin-module/spec.md, specs/1-ai-robot-brain-module/spec.md, specs/1-vision-language-action-module/spec.md updated.
- 🔁 Next prompts: Continue clarification process with remaining critical and high-priority questions.
- 🧠 Reflection: Successfully gathered initial critical clarifications from the user and updated relevant documentation. The next step is to continue with the remaining clarification points, prioritizing them as requested.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
