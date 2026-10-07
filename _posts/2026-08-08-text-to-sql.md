---
title: 'Text-to-SQL Assistant'
author: "Sree"
date: 2026-08-08 00:00:00
featured_image: '/images/blogs/12.text-to-sql/1.text-to-sql.jpg'
excerpt: Trusted Natural Language Analytics
description: "Turning Natural Language Questions into Trusted, Governed SQL"
tags: ["Agentic AI", "Text to SQL", "Marketing Analytics", "Data Science"]
categories: ["Machine Learning", "Generative AI"]
author_bio: "Sree is a Marketing Data Scientist and writer specializing in AI, analytics, and data-driven marketing."

---

<small style="margin-bottom: -10px; display: block;">
  *An excerpt from an article written by Sree, published in 'AI in Plain English' Journal.*
</small>


<style>
body {
text-align: justify}
</style>


Large language models have made generating SQL from natural language surprisingly easy.

But in an enterprise environment, generating SQL is only a small part of the problem.

The harder question is:

Can we trust the answer?

Over the past few years, I have worked on building and maintaining analytics products covering sales, marketing, customer experience and operational performance. As these datasets became more mature and widely used, a natural opportunity emerged: could we allow business users to interact with this data directly using plain English?

That led to the development of a series of governed, domain-specific Text-to-SQL assistants.

The objective was simple:

>"Allow business users to ask questions in plain English and receive fast, trusted answers from approved analytics datasets - without needing to write SQL."
>

This article walks through the thinking behind the solution, its architecture, the governance framework, how we evaluated it, and some of the most important lessons I learned along the way.

#### The Starting Point: A Mature Analytics Foundation

One of my core responsibilities at Appliances Online has been building analytics products that help monitor business performance across different time horizons - from yearly and monthly reporting all the way down to daily and near real-time hourly performance.

Over time, we developed data products bringing together areas such as:

1. Sales and revenue
2. Products and inventory
3. Profitability
4. Website traffic and conversion
5. Customer experience
6. Marketing performance
7. Customer behaviour
8. Operational metrics
This created a strong analytics foundation.

But it also created another challenge.

As access to data improved, people naturally started asking more questions.

![](/images/blogs/12.text-to-sql/2.text-to-sql.jpg)

#### The Business Problem
As the organisation became increasingly data-driven, demand for ad-hoc analysis grew.

A business user might ask:

* How is revenue tracking compared with last week?
* Which product categories are driving today's sales decline?
* Has conversion dropped for a particular state?
* Which products are close to being out of stock?
* How are different customer segments performing?

The data required to answer many of these questions already existed.

The challenge was accessibility.

Answering them often required someone who understood:

1. Where the relevant data lived.
2. Which tables and fields to use.
3. How the business defined the metric.
4. How to write the appropriate SQL.
5. How to interpret the result.
As a result, even relatively straightforward questions could become analyst requests.

The traditional workflow looked something like:

Business question * Analyst * SQL * Validation * Answer

This works, but it doesn't scale particularly well.

Analysts can gradually become an interface between business users and their own data.

That led to a simple question:

"What if a business user could ask the same question directly in plain English?"


#### Introducing Governed Text-to-SQL
The basic concept behind Text-to-SQL is straightforward.

A user asks:

"Show me revenue by category for the last seven days compared with the previous seven days."


A large language model interprets the question and generates the corresponding SQL.

BigQuery executes the query and the result is returned to the user in a business-friendly format.

Conceptually:

Business Question * LLM * SQL * BigQuery * Answer

But I quickly realised that this architecture alone wasn't sufficient for an enterprise analytics environment.

The model shouldn't have unrestricted access to the data warehouse.

And generating syntactically valid SQL doesn't necessarily mean generating the correct business answer.

We needed another layer:

governance.

#### Why I Chose Domain-Specific Agents
One of the important architectural decisions was not to build a single assistant capable of querying everything in the warehouse.

Instead, I divided the problem into business domains.

The initial framework consisted of three assistants:

##### Sales Performance
Focused on questions around:

* Revenue and sales trends
* Brands, products and categories
* Geographic performance
* Website traffic and conversion
* Sales funnel movements
* Out-of-stock and near-out-of-stock products

##### Customer Experience
Focused on areas such as:

* NPS performance
* Customer sentiment trends
* Delivery lead times
* Price competitiveness

##### Marketing Insights
Focused on:

* Customer acquisition and retention
* Customer segment movements
* Customer lifetime value
* Marketing-channel performance
* Subscriber growth
* Lead generation
This separation turned out to be useful for several reasons.

Each agent had a clearly defined responsibility.

It reduced the amount of data and metadata the model needed to understand at once.

And, importantly, it made the system easier to govern, evaluate and explain.

#### Architecture
I deliberately kept the architecture relatively simple.

For the Sales Performance assistant, the flow looked conceptually like this:

![](/images/blogs/12.text-to-sql/3.text-to-sql.jpg)

The agent wasn't simply given a database and told to generate SQL.

Before querying the data, it had access to contextual information including:

* Approved datasets
* Dataset and column descriptions
* KPI definitions
* Business terminology
* Semantic relationships
* Known business rules
* Verified query patterns
This context helped the model understand not only the structure of the data, but also what the data meant to the business.

That distinction is extremely important.


### About the Author

Sree is a Marketing Data Scientist and seasoned writer with over a decade of experience in data science and analytics, focusing on marketing and consumer analytics. Based in Australia, Sree is passionate about simplifying complex topics for a broad audience. His articles have appeared in esteemed outlets such as Towards Data Science, Generative AI, The Startup, and AI Advanced Journals. Learn more about his journey and work on his [portfolio - his digital home](https://srees.org/).

