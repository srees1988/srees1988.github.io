---
title: 'Evaluating Text-to-SQL Agents'
author: "Sree"
date: 2026-10-02 00:00:00
featured_image: '/images/blogs/12.eval-text-to-sql/1.eval-text-to-sql.jpg'
excerpt: Beyond SQL Correctness
description: "Why Correct SQL Doesn't Always Mean a Correct Answer?"
tags: ["Agentic AI", "Text to SQL", "Enterprise Analytics", "Data Science"]
categories: ["Machine Learning", "Generative AI"]
author_bio: "Sree is a Marketing Data Scientist and writer specializing in AI, analytics, and data-driven marketing."

---

<small style="margin-bottom: -10px; display: block;">
  *An excerpt from an article written by Sree, published in 'Generative AI' Journal.*
</small>


<style>
body {
text-align: justify}
</style>


Building a Text-to-SQL agent is relatively straightforward.

Evaluating whether you can **trust** it is much harder.

When I started working with governed Text-to-SQL assistants, one of the most important lessons was that successful SQL execution is a surprisingly poor measure of success.

A query can be:

*syntactically valid,
*execute successfully,
*return a perfectly reasonable-looking number,

and still give the **wrong business answer**.

That changed the way I thought about evaluating these systems.

Rather than asking:

**"Did the SQL execute?"**

I started with a much more important question:

**"Can we trust the business answer?"**

That led to a practical evaluation framework built around three dimensions:

**SQL Correctness * Answer Accuracy * Reliability**

### The Problem With "Valid SQL"

Imagine a business user asks:

**"Which category had the highest revenue growth last month?"**

The agent generates SQL.

The SQL executes successfully.

It returns: **Laundry: +12.4%**

Everything appears to be working.

But suppose the generated query:

*used order date instead of invoice date,
*included cancelled orders,
*used the wrong revenue calculation,
*or compared last month with the immediately preceding month when the business intended year-on-year growth.

The SQL is technically valid.

The database happily executes it.

The result may even look plausible.

But from a business perspective, the answer could be completely wrong.

This is why evaluating Text-to-SQL requires us to separate **technical correctness** from **business correctness**.


### 1. SQL Correctness

The first evaluation layer asks: Did the agent generate technically correct SQL?

There are several things we can inspect.

Did the agent:

*understand the user's request?
*generate valid SQL?
*select an approved dataset?
*use the correct tables and fields?
*construct the appropriate joins?
*apply the correct filters?
*use the right aggregations?

Consider a simple question:

**"What was revenue yesterday?"**

At the SQL level, we might check whether the query:

*references the expected sales dataset,
*uses the appropriate revenue field,
*applies a date filter,
*aggregates the result correctly,
*and executes without errors.

Many of these checks are relatively deterministic.

We can inspect the generated SQL, parse it, dry-run it or execute it against a controlled environment.

But this still doesn't tell us whether the resulting number is actually the number the business expects.

For that, we need another layer.

### 2. Answer Accuracy
This is where Text-to-SQL evaluation becomes much more interesting.

**Valid SQL does not necessarily mean a valid business answer.**

Answer accuracy asks:

Did the agent apply the correct business meaning and return the correct result?

For a sales analytics agent, that might require validating:

*Revenue definitions
*Order status
*Invoice status
*Date fields
*Customer definitions
*Comparison periods
*Inclusion and exclusion rules

Suppose our trusted reporting definition says that revenue should be based on invoiced transactions.

An agent could generate technically perfect SQL against an orders table.

Nothing is syntactically wrong.

But if that isn't how the organisation defines reported revenue, the answer is wrong.

This is why I find it useful to distinguish between:

**Syntactic correctness**

Can the database understand and execute the query?

and

**Semantic correctness**

Does the query represent what the business question actually means?

The second is usually much harder.


### Compare the Answer Against Something You Trust

One of the simplest ways to evaluate answer accuracy is to maintain trusted reference answers.

Depending on the question, these might come from:

*Analyst-written SQL
*Trusted analytical datasets
*Existing KPI reports
*Validated dashboards
*Known business results

Consider a small evaluation set:

![](/images/blogs/12.text-to-sql/2.eval-text-to-sql.jpg)

The third question is particularly interesting.

The SQL might execute successfully.

But why did the agent calculate **3.6% instead of 3.2%?**

Perhaps it used the wrong denominator.

That failure tells us much more than simply knowing whether the SQL executed.

### 3. Reliability

Even a system that produces the correct answer once isn't necessarily reliable.

Business users rarely phrase questions identically.

One person might ask:

"What was revenue yesterday?"

Another: "How much did we make yesterday?"

Another: "Show me yesterday's sales."

Depending on the organisation's terminology, those questions may or may not mean exactly the same thing.

A reliable agent needs to behave predictably across these variations.

So the third evaluation dimension asks whether the agent:

*produces consistent answers to paraphrased questions,
*follows approved business definitions,
*stays within approved datasets,
*respects semantic rules,
*handles ambiguous questions appropriately,
*and avoids answering questions outside its supported scope.

For example:

"How are sales looking?"

There may not be enough information to generate a meaningful query.

Which period?

Compared with what?

Revenue, orders or units?

Sometimes the best response isn't SQL.

It is:

"Which time period would you like to analyse?"

Knowing when not to generate SQL is part of reliability too.

![](/images/blogs/12.text-to-sql/3.eval-text-to-sql.jpg)


### Building a Representative Test Set
Once we define what "good" looks like, we need questions that genuinely test it.

I prefer representative business questions rather than artificial SQL benchmarks alone.

The test set can cover different levels of complexity:

*Simple aggregations
*Filters
*Rankings
*Trends
*Period comparisons
*Multiple dimensions
*Business-rule-dependent questions
*Ambiguous questions
*Unsupported questions

For example:

*"What was revenue yesterday?"
*"Which category had the highest revenue growth last month?"
*"Which brands declined compared with the same period last year?"
*"Which high-selling SKUs are currently low in inventory?"
*"How are sales looking?"

These questions test different parts of the agent.

The last one isn't necessarily testing SQL generation at all.

It is testing whether the agent recognises ambiguity.

### Turning Evaluation into a scorecard
The framework can then become a relatively simple scorecard.

![](/images/blogs/12.text-to-sql/4.eval-text-to-sql.jpg)

This gives us something much more useful than:

"The agent seems pretty good."

We can start measuring where it succeeds and, more importantly, where it fails.

### Not Every Evaluation Needs an LLM
Another useful lesson was that evaluation methods should match the type of behaviour being tested.

Some things are objective and should be tested deterministically.

Others require judgement.

This creates a hybrid evaluation framework.

For example:

![](/images/blogs/12.text-to-sql/5.eval-text-to-sql.jpg)

**Deterministic checks** are useful wherever there is an objective expected result.

**LLM-as-a-judge** can help with more subjective behaviours such as whether an ambiguous question was handled appropriately.

And **human review** remains useful for complex business semantics and periodically validating that the evaluation framework itself remains sensible.

I wouldn't replace deterministic tests with LLM judging where an exact comparison is possible.

If the trusted answer is **$4.8 million**, we don't need another LLM to tell us whether **$4.8 million equals $4.8 million**.


### Learning from Failures
The failures are arguably the most valuable part of the evaluation process.

Suppose an evaluation suite contains 200 questions:

**192 pass . 8 fail. Pass rate: 96%**

The interesting part isn't necessarily the 96%.

It's the eight failures.

Why did they fail?

Perhaps the agent:

*chose the wrong join,
*misunderstood a KPI,
*used an incorrect date field,
*misunderstood internal terminology,
*lacked an important business definition,
*or encountered an ambiguous question.

Each failure points towards something that can be improved.

Depending on the root cause, we might refine:

*Dataset metadata
*Business glossary
*Semantic definitions
*Agent instructions
*Guardrails
*Verified query examples

Then we run the tests again.

The evaluation process therefore becomes a continuous loop:

**Define * Test * Score * Diagnose * Improve * Re-test**

### From Manual Testing to Automated Regression Testing

Early in development, this can be fairly manual.

Ask a question.

Inspect the generated SQL.

Run it.

Compare the result.

Record whether it passed.

That works when there are 20 questions.

It becomes painful when there are 200.

And it becomes even more difficult when every change to a prompt, semantic layer, model or business definition requires running those tests again.

That's where automated regression testing becomes useful.

Conceptually:

![](/images/blogs/12.text-to-sql/6.eval-text-to-sql.jpg)

The same approved evaluation set can then be rerun whenever something important changes.

For example:

**Before change: 192 / 200 passed**

A semantic definition is modified.

Run the suite again.

**After change: 196 / 200 passed**

But we shouldn't look only at the overall number.

Did the four previously failing questions improve?

Did any previously passing questions now fail?

That second question is crucial.

A change that fixes one behaviour can inadvertently break another.

In that sense, evaluating an AI agent starts to look surprisingly similar to **regression testing in software engineering.**

### The Metric That Matters Most
It's tempting to evaluate a Text-to-SQL system based on whether it generates impressive-looking SQL.

But business users don't actually care about SQL.

They care about the answer.

And ultimately they care about whether they can **trust that answer enough to use it in a decision.**

That's why I think about Text-to-SQL evaluation as three increasingly important layers:

* **SQL Correctness** (Did we generate the right query?)


* **Answer Accuracy**  (Did we produce the right business result?)


* **Reliability** (Can we trust the system to behave consistently over time?)

The SQL is only an intermediate artefact.

**Trust is the real output.**



### About the Author

Sree is a Marketing Data Scientist and seasoned writer with over a decade of experience in data science and analytics, focusing on marketing and consumer analytics. Based in Australia, Sree is passionate about simplifying complex topics for a broad audience. His articles have appeared in esteemed outlets such as Towards Data Science, Generative AI, The Startup, and AI Advanced Journals. Learn more about his journey and work on his [portfolio - his digital home](https://srees.org/).


--------------------------------------------

Large language models have made generating SQL from natural language surprisingly easy.

But in an enterprise environment, generating SQL is only a small part of the problem.

The harder question is:

Can we trust the answer?

Over the past few years, I have worked on building and maintaining analytics products covering sales, marketing, customer experience and operational performance. As these datasets became more mature and widely used, a natural opportunity emerged: could we allow business users to interact with this data directly using plain English?

That led to the development of a series of governed, domain-specific Text-to-SQL assistants.

The objective was simple:

**"Allow business users to ask questions in plain English and receive fast, trusted answers from approved analytics datasets - without needing to write SQL."**


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

**Business question * Analyst * SQL * Validation * Answer**

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

**Business Question * LLM * SQL * BigQuery * Answer**

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


**Valid SQL Doesn't Always Mean a Valid Answer**

Consider a deceptively simple question:

**"What was revenue yesterday?"**

A model might easily generate perfectly valid SQL. But several business questions remain.

* What does revenue mean?
* Should it use order date or invoice date?
* Should cancelled transactions be included?
* Should refunds be deducted?
* Should GST be included?


The SQL could execute successfully while still producing the wrong business number.

This was one of the most important lessons from the project:

**Text-to-SQL isn't only a SQL-generation problem. It is a business-semantics problem.**

That is why business definitions and the semantic layer became central components of the architecture.


#### Building Governance Around the LLM

Because the assistants were designed for business users, trust and consistency were critical.

Several safeguards were therefore built around the model.

**1. Approved datasets only**
Agents could query only datasets that had already been validated and were trusted for analytics.

They did not have unrestricted access to the wider warehouse.

**2. Business definitions**
Important metrics and terminology were explicitly defined.

The model shouldn't have to guess what terms such as revenue, conversion, active customer or returning customer mean within the organisation.

**3. Semantic rules**
Relationships between dimensions, metrics and datasets were provided as context so that queries followed established business logic.

**4. Read-only access**
The assistant was designed for analytics.

Generated SQL therefore operated under read-only principles rather than allowing the model to modify underlying data.

**5. Query-cost controls**
Large analytical queries can become expensive.

Controls were introduced to restrict the amount of data a generated query could process - for example, enforcing a maximum processing threshold of approximately 2 GB for applicable queries.

**6. Human oversight**
AI did not eliminate the analyst.

Complex or unusual questions could still be reviewed, and outputs could be validated against existing dashboards, KPI reports and analyst-written queries.

The objective was not:

**AI instead of analysts.**

It was:

**AI for routine questions, analysts for higher-value investigation and decision support.**

#### How Do You Evaluate a Text-to-SQL Agent?

Once the system was working, another interesting question emerged:

How do we know whether it is actually good?

Simply checking whether the SQL executes isn't enough.

I eventually thought about evaluation across three main dimensions.

**1. SQL Correctness**

Did the agent generate the appropriate SQL logic?

This includes areas such as:

* Correct tables
* Correct joins
* Correct filters
* Correct aggregations
* Correct date logic
* Correct business rules

**2. Answer Accuracy**

Even if the SQL looks reasonable, does the resulting answer match the trusted business result?

Outputs could be compared against:

* Analyst-written SQL
* Existing trusted dashboards
* Known KPI results
* Validated reporting datasets

This distinction matters.

SQL correctness asks whether the query is right.

Answer accuracy asks whether the business answer is right.

**3. Reliability**

Finally, could the agent continue producing the correct result when:

* A question was phrased differently?
* The date range changed?
* Different dimensions were requested?
* Multiple metrics were combined?

A useful analytics assistant needs to be more than occasionally impressive.

It needs to be consistently dependable.

#### Moving from Manual Evaluation to Regression Testing

Initially, evaluation involved representative business questions and manual comparison against trusted results.

When failures occurred, I looked at where they originated.

Was the problem:

* An unclear business definition?
* Missing metadata?
* An incorrect semantic relationship?
* Ambiguous prompting?
* A SQL-generation error?

The appropriate layer could then be refined.

Over time, these representative questions became reusable evaluation sets.

That opened the door to automated regression testing.

Changes to prompts, models, semantic definitions or agent logic could be tested against known questions before being released.

This became particularly important as the system expanded beyond the first Sales Performance assistant into additional business domains.


#### Business Impact
There are two ways I think about the potential value of this type of system.

**Reducing repetitive analyst workload**

Consider a relatively conservative scenario.

Suppose approximately 25 routine analytics questions are raised each week and each requires around 30 minutes of analyst effort.

That represents:

**25 questions * 30 minutes * 52 weeks = 650 analyst hours per year**

At an illustrative analytics cost of AUD 80-100 per hour, that equates to roughly AUD 50,000-65,000 of annual analyst capacity that could potentially be redirected toward higher-value work.

The important point isn't necessarily the dollar figure.

It is what those hours can become.

Instead of repeatedly answering:

"What was revenue yesterday?"

an analyst can spend more time answering:

"Why is revenue changing, and what should we do about it?"

#### The Bigger Opportunity: Reducing Time-to-Insight

For me, the more interesting value was not simply productivity.

It was decision speed.

Imagine sales suddenly decline by 10%.

The metric may already exist somewhere in a dashboard.

But identifying what changed could require someone to investigate:

* Which category declined?
* Which state was affected?
* Was traffic down?
* Did conversion change?
* Was there an inventory problem?
* Was the movement concentrated in particular products?

Traditionally, that can trigger an analyst request and investigation.

With a governed conversational analytics layer, a leader can begin interrogating those questions immediately.

For a business operating at significant daily revenue volumes, even relatively small improvements in the speed at which material issues are identified and investigated can have meaningful commercial value.

For example, in a hypothetical business generating AUD 5 million  per day, 1% of daily revenue represents AUD 50,000.

That doesn't mean a Text-to-SQL assistant automatically "creates" or "saves" that revenue.

But it illustrates why reducing the time between **signal * investigation * action** can matter.

#### What I Learned

The biggest lesson from this project was surprisingly simple.

Generating SQL wasn't the hardest part. Generating trust was.

Modern language models can generate impressive SQL remarkably quickly.

But enterprise analytics requires more.

The things that ultimately mattered most were:

* Clean and trusted datasets
* Clear business definitions
* Strong semantic context
* Well-defined agent boundaries
* Query guardrails
* Consistent evaluation
* Human oversight where appropriate

Model capability certainly matters.

But once the model is reasonably capable, improvements in governance, metadata and business semantics can sometimes be more valuable than simply switching to a more sophisticated model.

That changed how I think about enterprise AI systems.

The goal isn't to build the most impressive AI demo.

The goal is to build something that people can confidently use to make decisions.

For me, that comes down to three principles:

**Trust > Technology**

**Governance > Model Sophistication**

**Business Value > AI Novelty**



