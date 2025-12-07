---
sidebar_position: 5
title: "Submission Guidelines"
---

# Submission Guidelines

This document outlines the requirements and process for submitting your Autonomous Humanoid capstone project. Follow these guidelines carefully to ensure your submission meets all requirements and can be properly evaluated.

## Deliverables Checklist

Your capstone project submission must include the following components:

### Required Deliverables
- [ ] **Source Code Repository** - Complete ROS 2 workspace with all packages
- [ ] **Technical Documentation** - Code comments, API documentation, and system architecture
- [ ] **Demonstration Video** - 3-5 minute video showing system functionality
- [ ] **Technical Report** - 5-10 page report detailing implementation and results
- [ ] **Configuration Files** - All launch files, parameter files, and setup scripts
- [ ] **Test Results** - Output from unit and integration tests
- [ ] **Performance Metrics** - Collected metrics showing system performance

### Optional Enhancements (Bonus Points)
- [ ] **Advanced Features** - Multi-modal interaction, learning capabilities
- [ ] **Performance Optimizations** - Real-time processing, energy efficiency
- [ ] **Extended Functionality** - Multi-robot coordination, cloud integration
- [ ] **Additional Scenarios** - Complex task execution beyond basic requirements

## GitHub Repository Structure

Organize your GitHub repository using the following structure:

```
autonomous-humanoid-project/
├── README.md                    # Project overview and setup instructions
├── .gitignore                  # Git ignore file for ROS 2 projects
├── docs/                       # Additional documentation
│   ├── architecture.md         # System architecture documentation
│   ├── user_guide.md           # User manual for the system
│   └── performance.md          # Performance analysis and metrics
├── src/                        # ROS 2 source code
│   ├── voice_interface/        # Voice processing package
│   ├── task_planner/           # LLM task planning package
│   ├── navigation_controller/  # Navigation system package
│   ├── perception_module/      # Perception system package
│   ├── manipulation_controller/ # Manipulation system package
│   └── integration_layer/      # System integration package
├── test/                       # Test files and scripts
├── config/                     # Configuration and parameter files
├── launch/                     # Launch files for system startup
├── scripts/                    # Utility scripts for setup and testing
├── results/                    # Test results and performance metrics
└── report/                     # Technical report files
    ├── report.pdf             # Final technical report
    ├── presentation.pdf       # Project presentation slides
    └── bibliography.bib       # Reference bibliography
```

### Repository Setup Instructions

1. **Create Repository**: Create a new private repository on GitHub named `autonomous-humanoid-project`
2. **Initialize**: Clone the repository and set up the basic structure above
3. **Add Files**: Copy your completed ROS 2 workspace into the `src/` directory
4. **Documentation**: Add comprehensive documentation in the `docs/` directory
5. **Results**: Include all test results and performance metrics in the `results/` directory
6. **Report**: Place your final report in the `report/` directory
7. **Commit**: Commit all changes with descriptive commit messages
8. **Push**: Push the repository to GitHub

## Video Demonstration Requirements

Create a 3-5 minute video demonstrating your system's functionality:

### Video Content Requirements
- **Duration**: 3-5 minutes (strictly enforced)
- **Introduction**: Brief project overview (30 seconds)
- **System Demo**: Show all required functionality (2-3 minutes)
- **Technical Discussion**: Explain key implementation aspects (30 seconds - 1 minute)
- **Results Summary**: Highlight performance metrics and achievements (30 seconds)

### Video Technical Requirements
- **Resolution**: Minimum 720p (1280x720), recommended 1080p (1920x1080)
- **Format**: MP4, MOV, or AVI format
- **Audio**: Clear audio for voice-over and system sounds
- **Quality**: Good lighting and stable camera work
- **Editing**: Smooth transitions between scenes, no excessive effects

### Required Demonstration Scenarios
Your video must show the system successfully completing these scenarios:

1. **Simple Object Fetch**: "Robot, please bring me the red cup from the kitchen counter."
   - Voice command processing
   - Task planning and decomposition
   - Navigation to kitchen
   - Object detection and identification
   - Grasping and manipulation
   - Return to user

2. **Multi-Step Task**: "Clean up the living room by putting books on the shelf and disposing of the trash."
   - Complex command interpretation
   - Multi-task planning
   - Sequential execution
   - Task completion reporting

3. **Adaptive Behavior**: "Move the green box to the blue table, but if the table is occupied, place it on the floor next to the table."
   - Environmental awareness
   - Conditional execution
   - Adaptive planning

### Video Editing Guidelines
- **Keep it concise**: Focus on key functionality rather than extended setup
- **Highlight success**: Show successful completions rather than failed attempts
- **Add annotations**: Use text overlays to highlight important moments
- **Include audio**: Narrate the demonstration to explain what's happening
- **Show diversity**: Include different lighting conditions and environments

## Technical Report Outline

Your technical report should be 5-10 pages and follow this structure:

### 1. Abstract (150-200 words)
- Brief summary of the project
- Key technologies used
- Main achievements and results
- Performance metrics summary

### 2. Introduction (0.5-1 page)
- Project motivation and objectives
- Problem statement
- Approach and methodology
- Report organization

### 3. System Architecture (1-2 pages)
- High-level system design
- Component interactions and data flow
- Technology stack and tools
- Design decisions and rationale

### 4. Implementation (2-3 pages)
- Detailed implementation approach
- Key algorithms and techniques
- Code structure and organization
- Challenges encountered and solutions

### 5. Results and Evaluation (1-2 pages)
- Test methodology and scenarios
- Performance metrics and analysis
- Comparison with requirements
- Successes and limitations

### 6. Challenges and Solutions (0.5-1 page)
- Technical challenges faced
- Solutions implemented
- Lessons learned
- Future improvements

### 7. Conclusion (0.5 page)
- Summary of achievements
- Key takeaways
- Future work recommendations

### 8. References
- Properly formatted citations
- Academic papers, documentation, and sources used
- Follow IEEE or ACM citation style

### Report Formatting Requirements
- **Length**: 5-10 pages (excluding references)
- **Format**: PDF format
- **Font**: Times New Roman 12pt or equivalent
- **Spacing**: Double-spaced text
- **Margins**: 1 inch on all sides
- **Figures**: High-quality images and diagrams with captions
- **Tables**: Properly formatted with titles and units

## Grading Criteria

Your project will be evaluated across five key areas with the following weightings:

| Criteria | Weight | Description | Evaluation Points |
|----------|--------|-------------|-------------------|
| Integration | 30% | How well modules 1-4 work together | • Component communication (10%) <br />• System architecture (10%) <br />• Cross-module functionality (10%) |
| Technical Implementation | 30% | Code quality, architecture, and functionality | • Code quality and documentation (10%) <br />• Algorithm implementation (10%) <br />• System functionality (10%) |
| Robustness | 20% | Error handling, adaptability, and reliability | • Error handling (10%) <br />• System reliability (10%) |
| Innovation | 10% | Creative solutions and novel approaches | • Novel approaches (5%) <br />• Bonus features (5%) |
| Documentation | 10% | Code comments, user guides, and technical report | • Code documentation (5%) <br />• Technical report quality (5%) |

### Detailed Rubric

#### Integration (30 points)
- **Excellent (27-30)**: Seamless integration across all modules, sophisticated inter-component communication, advanced architectural patterns
- **Good (24-26)**: Good integration with minor issues, clear component interactions, proper ROS 2 practices
- **Satisfactory (21-23)**: Basic integration working, some communication issues, standard practices followed
- **Needs Improvement (18-20)**: Integration issues present, communication problems, poor architectural decisions
- **Unsatisfactory (0-17)**: Poor or non-existent integration, fundamental communication failures

#### Technical Implementation (30 points)
- **Excellent (27-30)**: High-quality, well-documented code, sophisticated algorithms, exceptional functionality
- **Good (24-26)**: Good quality code, solid implementation, most functionality working well
- **Satisfactory (21-23)**: Adequate code quality, basic functionality implemented
- **Needs Improvement (18-20)**: Code quality issues, missing functionality, poor implementation
- **Unsatisfactory (0-17)**: Poor code quality, significant functionality missing

#### Robustness (20 points)
- **Excellent (18-20)**: Exceptional error handling, highly reliable, adapts well to variations
- **Good (16-17)**: Good error handling, reliable operation, good adaptability
- **Satisfactory (14-15)**: Basic error handling, generally reliable
- **Needs Improvement (12-13)**: Limited error handling, reliability issues
- **Unsatisfactory (0-11)**: Poor error handling, unreliable system

#### Innovation (10 points)
- **Excellent (9-10)**: Highly innovative solutions, significant bonus features implemented
- **Good (7-8)**: Good innovative elements, some bonus features
- **Satisfactory (5-6)**: Some innovative thinking, basic bonus features
- **Needs Improvement (3-4)**: Limited innovation, few bonus features
- **Unsatisfactory (0-2)**: No innovation, no bonus features

#### Documentation (10 points)
- **Excellent (9-10)**: Comprehensive, well-organized documentation, exceptional report
- **Good (7-8)**: Good documentation, quality report
- **Satisfactory (5-6)**: Adequate documentation, acceptable report
- **Needs Improvement (3-4)**: Limited documentation, poor report
- **Unsatisfactory (0-2)**: Poor documentation, inadequate report

## Submission Process

### Step 1: Final Testing
Before submission, ensure your system passes all tests:
1. Run all unit tests and verify they pass
2. Execute all integration test scenarios
3. Verify performance metrics meet requirements
4. Test all required demonstration scenarios

### Step 2: Repository Preparation
1. Clean up your repository (remove build artifacts, temporary files)
2. Ensure all code is properly committed
3. Update README with clear setup and usage instructions
4. Verify all required files are included

### Step 3: Video Creation
1. Plan your demonstration scenarios
2. Record high-quality video showing all requirements
3. Edit video to meet specifications
4. Upload to appropriate platform (YouTube, Vimeo, etc.)

### Step 4: Report Completion
1. Write comprehensive technical report following outline
2. Include all required sections and figures
3. Verify formatting requirements are met
4. Proofread for clarity and correctness

### Step 5: Final Submission
1. Push all changes to your GitHub repository
2. Ensure repository is accessible to evaluators
3. Submit repository URL through course management system
4. Include video link and any additional submission materials

## Late Submission Policy

- **1-2 days late**: 5% grade reduction
- **3-7 days late**: 10% grade reduction
- **More than 7 days late**: Not accepted without prior approval

## Additional Resources

### Reference Materials
- [Module 1: ROS 2 Documentation](../module-1-ros2/)
- [Module 2: Digital Twin Documentation](../module-2-digital-twin/)
- [Module 3: AI-Robot Brain Documentation](../module-3-ai-robot-brain/)
- [Module 4: Vision-Language-Action Documentation](../module-4-vision-language-action/)

### Support Channels
- Course staff office hours
- Discussion forums
- Technical support for hardware issues
- ROS 2 community resources

### Best Practices
- Start early and work incrementally
- Test each component thoroughly before integration
- Maintain good version control practices
- Document your work as you develop
- Seek help when encountering persistent issues