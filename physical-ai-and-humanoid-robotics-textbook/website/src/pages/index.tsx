import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroContainer}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>🤖 Physical AI Education</div>
          <Heading as="h1" className={styles.heroTitle}>
            {siteConfig.title}
          </Heading>
          <p className={styles.heroSubtitle}>
            Master the future of robotics with hands-on training in ROS 2, simulation, 
            NVIDIA Isaac, and Vision-Language-Action models
          </p>
          <div className={styles.heroButtons}>
            <Link
              className={clsx('button button--primary button--lg', styles.primaryButton)}
              to="/docs/intro/">
              Start Learning
            </Link>
            <Link
              className={clsx('button button--outline button--lg', styles.secondaryButton)}
              to="/docs/getting-started/hardware-requirements">
              View Prerequisites
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function ModulesSection() {
  const modules = [
    {
      title: 'Module 1: ROS 2',
      icon: '⚙️',
      description: 'Master the Robot Operating System - the nervous system of modern robots',
      link: '/docs/module-1-ros2/',
    },
    {
      title: 'Module 2: Digital Twin',
      icon: '🌐',
      description: 'Build realistic simulations with Gazebo and Unity for robot testing',
      link: '/docs/module-2-digital-twin/',
    },
    {
      title: 'Module 3: AI-Robot Brain',
      icon: '🧠',
      description: 'Leverage NVIDIA Isaac for perception, navigation, and intelligent planning',
      link: '/docs/module-3-ai-robot-brain/',
    },
    {
      title: 'Module 4: Vision-Language-Action',
      icon: '👁️',
      description: 'Integrate voice commands and LLMs for natural human-robot interaction',
      link: '/docs/module-4-vision-language-action/',
    },
  ];

  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>
            Four Comprehensive Modules
          </Heading>
          <p className={styles.sectionSubtitle}>
            A structured journey from fundamentals to advanced embodied AI
          </p>
        </div>
        <div className={styles.moduleGrid}>
          {modules.map((module, idx) => (
            <Link to={module.link} className={styles.moduleCard} key={idx}>
              <div className={styles.moduleIcon}>{module.icon}</div>
              <Heading as="h3" className={styles.moduleTitle}>
                {module.title}
              </Heading>
              <p className={styles.moduleDescription}>{module.description}</p>
              <div className={styles.moduleArrow}>→</div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    {
      icon: '📚',
      title: 'Comprehensive Content',
      description: '40+ chapters covering theory and practical implementation',
    },
    {
      icon: '💻',
      title: 'Hands-On Tutorials',
      description: 'Real code examples and step-by-step guides for every concept',
    },
    {
      icon: '🎯',
      title: 'Project-Based Learning',
      description: 'Build a complete autonomous humanoid in the capstone project',
    },
    {
      icon: '🚀',
      title: 'Industry-Relevant',
      description: 'Learn tools used by leading robotics companies worldwide',
    },
  ];

  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className={styles.featuresGrid}>
          {features.map((feature, idx) => (
            <div className={styles.featureCard} key={idx}>
              <div className={styles.featureIcon}>{feature.icon}</div>
              <Heading as="h3" className={styles.featureTitle}>
                {feature.title}
              </Heading>
              <p className={styles.featureDescription}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className={styles.ctaContent}>
          <Heading as="h2" className={styles.ctaTitle}>
            Ready to Build the Future of Robotics?
          </Heading>
          <p className={styles.ctaDescription}>
            Join the next generation of roboticists and AI engineers shaping embodied intelligence
          </p>
          <Link
            className={clsx('button button--primary button--lg', styles.ctaButton)}
            to="/docs/intro/">
            Begin Your Journey
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="A comprehensive textbook on Physical AI and Humanoid Robotics covering ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action models.">
      <HomepageHeader />
      <main>
        <ModulesSection />
        <FeaturesSection />
        <CTASection />
      </main>
    </Layout>
  );
}
