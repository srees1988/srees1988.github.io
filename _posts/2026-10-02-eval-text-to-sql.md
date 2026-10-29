---
title: 'Evaluating Text-to-SQL Agents'
author: "Sree"
date: 2026-10-02 00:00:00
featured_image: '/images/blogs/13.eval-text-to-sql/1.eval-text-to-sql.png'
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


Building a Text-to-SQL agent is relatively straightforward. Evaluating whether you can **trust** it is much harder.

When I started working with governed Text-to-SQL assistants, one of the most important lessons was that successful SQL execution is a surprisingly poor measure of success.

A query can be:

* syntactically valid
* execute successfully
* return a perfectly reasonable-looking number

and still give the **wrong business answer**.

That changed the way I thought about evaluating these systems.

Rather than asking:

**"Did the SQL execute?"**

I started with a much more important question:

**"Can we trust the business answer?"**

That led to a practical evaluation framework built around three dimensions:

(1) **SQL Correctness** (2) **Answer Accuracy** (3) **Reliability**

### The Problem With "Valid SQL"

Imagine a business user asks:

**"Which category had the highest revenue growth last month?"**

The agent generates SQL. The SQL executes successfully. 

It returns: **Laundry: +12.4%**

Everything appears to be working. But suppose the generated query:

* Used order date instead of invoice date 
* Included cancelled orders
* Used the wrong revenue calculation
* or compared wrong date period

The SQL is technically valid. The database happily executes it. 

The result may even look plausible. But from a business perspective, the answer could be completely wrong. 

This is why evaluating Text-to-SQL requires us to separate **technical correctness** from **business correctness**.


### 1. SQL Correctness

The first evaluation layer asks: Did the agent generate technically correct SQL?

There are several things we can inspect.

Did the agent:

* Understand the user's request?
* Generate valid SQL?
* Select an approved dataset?
* Use the correct tables and fields?
* Construct the appropriate joins?
* Apply the correct filters?
* Use the right aggregations?

Consider a simple question:

**"What was revenue yesterday?"**

At the SQL level, we might check whether the query:

* References the expected sales dataset
* Uses the appropriate revenue field
* Applies a date filter
* Aggregates the result correctly
* and executes without errors.

Many of these checks are relatively deterministic. We can inspect the generated SQL, parse it, dry-run it or execute it against a controlled environment.

But this still doesn't tell us whether the resulting number is actually the number the business expects.

For that, we need another layer.

### 2. Answer Accuracy
This is where Text-to-SQL evaluation becomes much more interesting.

**Valid SQL does not necessarily mean a valid business answer.**

Answer accuracy asks:

Did the agent apply the correct business meaning and return the correct result?

For a sales analytics agent, that might require validating:

* Revenue definitions
* Order status
* Invoice status
* Date fields
* Customer definitions
* Comparison periods
* Inclusion and exclusion rules

Suppose our trusted reporting definition says that revenue should be based on invoiced transactions.

An agent could generate technically perfect SQL against an orders table.

Nothing is syntactically wrong. But if that isn't how the organisation defines reported revenue, the answer is wrong.

This is why I find it useful to distinguish between:

**Syntactic correctness** (Can the database understand and execute the query?)

and

**Semantic correctness** (Does the query represent what the business question actually means?)

The second is usually much harder.


### Compare the Answer Against Something You Trust

One of the simplest ways to evaluate answer accuracy is to maintain trusted reference answers.

Depending on the question, these might come from:

* Analyst-written SQL
* Trusted analytical datasets
* Existing KPI reports
* Validated dashboards
* Known business results

Consider a small evaluation set:

![](/images/blogs/13.eval-text-to-sql/2.eval-text-to-sql.png)

The third question is particularly interesting. The SQL might execute successfully. But why did the agent calculate **3.6% instead of 3.2%?**

Perhaps it used the wrong denominator.

That failure tells us much more than simply knowing whether the SQL executed.

### 3. Reliability

Even a system that produces the correct answer once isn't necessarily reliable. Business users rarely phrase questions identically.

One person might ask:

What was revenue yesterday?

Another: How much did we make yesterday?

Another: Show me yesterday's sales.

Depending on the organisation's terminology, those questions may or may not mean exactly the same thing.

A reliable agent needs to behave predictably across these variations.

So the third evaluation dimension asks whether the agent:

* Produces consistent answers to paraphrased questions
* Follows approved business definitions
* Stays within approved datasets
* Respects semantic rules
* Handles ambiguous questions appropriately
* and avoids answering questions outside its supported scope.

For example:

"How are sales looking?"

There may not be enough information to generate a meaningful query.

Which period?

Compared with what?

Revenue, orders or units?

Sometimes the best response isn't SQL. It is:

"Which time period would you like to analyse?"

Knowing when not to generate SQL is part of reliability too.

![](/images/blogs/13.eval-text-to-sql/3.eval-text-to-sql.png)


### Building a Representative Test Set
Once we define what "good" looks like, we need questions that genuinely test it.

I prefer representative business questions rather than artificial SQL benchmarks alone.

The test set can cover different levels of complexity:

* Simple aggregations
* Filters
* Rankings
* Trends
* Period comparisons
* Multiple dimensions
* Business-rule-dependent questions
* Ambiguous questions
* Unsupported questions

For example:

"What was revenue yesterday?"
"Which category had the highest revenue growth last month?"
"Which brands declined compared with the same period last year?"
"Which high-selling SKUs are currently low in inventory?"
"How are sales looking?"

These questions test different parts of the agent.

The last one isn't necessarily testing SQL generation at all.

It is testing whether the agent recognises ambiguity.

### Turning Evaluation into a scorecard
The framework can then become a relatively simple scorecard.

![](/images/blogs/13.eval-text-to-sql/4.eval-text-to-sql.png)

This gives us something much more useful than:

"The agent seems pretty good."

We can start measuring where it succeeds and, more importantly, where it fails.

### Not Every Evaluation Needs an LLM
Another useful lesson was that evaluation methods should match the type of behaviour being tested.

Some things are objective and should be tested deterministically. Others require judgement. This creates a hybrid evaluation framework.

For example:

![](/images/blogs/13.eval-text-to-sql/5.eval-text-to-sql.png)

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

* Chose the wrong join
* Misunderstood a KPI
* Used an incorrect date field
* Misunderstood internal terminology
* Lacked an important business definition
* or encountered an ambiguous question.

Each failure points towards something that can be improved.

Depending on the root cause, we might refine:

* Dataset metadata
* Business glossary
* Semantic definitions
* Agent instructions
* Guardrails
* Verified query examples

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

![](/images/blogs/13.eval-text-to-sql/6.eval-text-to-sql.png)

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

But business users don't actually care about SQL. They care about the answer.

And ultimately they care about whether they can **trust that answer enough to use it in a decision.**

That's why I think about Text-to-SQL evaluation as three increasingly important layers:

**SQL Correctness** (Did we generate the right query?)


**Answer Accuracy**  (Did we produce the right business result?)


**Reliability** (Can we trust the system to behave consistently over time?)

The SQL is only an intermediate artefact.

**Trust is the real output.**


### About the Author

Sree is a Marketing Data Scientist and seasoned writer with over a decade of experience in data science and analytics, focusing on marketing and consumer analytics. Based in Australia, Sree is passionate about simplifying complex topics for a broad audience. His articles have appeared in esteemed outlets such as Towards Data Science, Generative AI, The Startup, and AI Advanced Journals. Learn more about his journey and work on his [portfolio - his digital home](https://srees.org/).


