# Worldsmith

## Project Overview

Worldsmith is an experimental project exploring whether LLM agents can autonomously develop and improve executable representations of physical environments through interaction and experimentation.

The central idea is to represent a simulated world as code: physical objects become software objects, observations become variables, and interactions with the environment are expressed through executable programs.

Rather than manually implementing every algorithm needed to understand and interact with the world, Worldsmith explores whether an LLM can discover missing capabilities and develop them as reusable code.

## Core Concept

Worldsmith runs one shared Python controller. It receives the task name and observations at runtime, with no runtime LLM. Interception, moving-obstacle navigation, rendezvous / tracking, bouncing-object interception, and object pushing form the current benchmark. AutoResearch evaluates them together and improves representations and skills that the shared controller can reuse across tasks. The researcher can see task descriptions and development traces.

### Long-term vision

The project separates two complementary agents:

**1. Runtime Agent**

An LLM that receives a task and a structured representation of the environment. It generates executable code to interact with the simulated world using the available objects, functions, and skills.

**2. AutoResearch Agent**

A coding agent that improves the underlying world representation and its capabilities through experimentation.

It analyzes failures, identifies limitations, proposes new abstractions or algorithms, and evaluates whether its changes improve performance.

For example, the agent might discover that predicting object motion requires velocity estimation, or that reliable trajectory planning requires accounting for uncertainty.

These capabilities are not necessarily provided in advance. The goal is to allow the agent to discover and implement them.

## Learning Through Experimentation

Worldsmith follows a self-improving research loop:

1. Evaluate the current system across simulated tasks.
2. Observe failures, limitations, and performance.
3. Propose changes to the world model, functions, or skills.
4. Test the changes through simulation.
5. Retain improvements and repeat.

The learning process focuses on evolving executable code rather than updating the LLM's weights.

## Research Goals

The project explores several questions:

- Can an LLM identify missing abstractions or algorithms from environmental feedback?
- Can it develop useful representations of complex physical phenomena without those representations being explicitly programmed?
- Can discovered functions generalize across different environments and tasks?
- Does an evolving library of executable knowledge improve the performance of a code-generating agent over time?

A key distinction is between learning **how to perform a specific task** and learning **reusable computational models of how the world works**.

Worldsmith is particularly interested in the latter.

## Project Scope

Worldsmith is a personal research playground, not a production robotics system.

The initial focus is simulated environments, with an emphasis on experimentation, observable learning progress, and discovering interesting emergent capabilities.

The project should remain flexible enough to explore increasingly complex environments and new approaches to self-improving agent systems.

**The long-term vision is an agent that doesn't just learn to act within a world, but learns to write the code it needs to understand that world.**
