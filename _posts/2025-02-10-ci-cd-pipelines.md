---
title: 'Scaling Machine Learning'
date: 2025-02-10 00:00:00
featured_image: '/images/blogs/11.ci-cd-pipeline/1.ci-cd-pipeline.jpg'
excerpt: A Beginner's Guide to ML Automation

---


<style>
body {
text-align: justify}
</style>

![](/images/blogs/11.ci-cd-pipeline/1.ci-cd-pipeline.jpg)


Machine learning is transforming businesses in ways we never thought possible, especially in marketing and consumer analytics. It's all about using data-driven insights and predictions to make smarter decisions. But here's the thing - while many of us learn the basics of ML from courses and tutorials, they rarely cover the practical side, like setting up Continuous Integration and Deployment (CI/CD) pipelines. And honestly, those pipelines are what make it possible to scale ML models and solve real-world problems.

In this blog, I'll explain what CI/CD pipelines are using a simple analogy: building and delivering cars. Just like an assembly line makes car production smooth and efficient, CI/CD pipelines help streamline the process of creating and deploying machine learning models. I hope this serves as a practical, beginner-friendly guide to CI/CD pipelines in machine learning workflows for data scientists, analysts, and anyone curious about the topic.


#### What Is a CI/CD Pipeline and why does it matter?

Let's start with the basics: what is a CI/CD pipeline?

At its core, a CI/CD pipeline is a set of automated processes that help you build, test, and deploy machine learning models efficiently.

Imagine you're building a car on an assembly line. Every step - from sourcing raw materials, assembling parts, testing the vehicle, to rolling it off the line - is carefully automated to ensure everything works as it should. CI/CD pipelines do the same for machine learning projects:

- **CI (Continuous Integration):** This is like assembling the car's parts in the factory. Each time you add a new part (or code change), it's tested to make sure everything fits and works together seamlessly.

- **CD (Continuous Deployment):** Think of this as delivering the finished car to the dealership. It ensures your model is automatically deployed into production without any hiccups, ready to deliver value.

In simpler terms, CI/CD pipelines are like an assembly line for your ML projects. They take care of the repetitive tasks so you can focus on what really matters - creating impact.

Why does understanding and implementing CI/CD pipelines matter? It helps you:

a) Scale your machine learning models.

b) Solve tough business problems faster and more effectively.

c) Deliver real value to your team and stakeholders with smarter, data-driven solutions.

Honestly, these are the kind of lessons I wish someone had shared with me earlier in my career. And I hope this walkthrough gives you a clear and practical understanding of how to build and scale ML systems effectively on the cloud.


#### A Three-Pipeline Framework for CI/CD in Machine Learning

When it comes to real-world projects, a solid CI/CD setup needs three key pipelines working together seamlessly


- **Initial Training Pipeline:** The starting point where models are built and trained using historical data.

- **Prediction Pipeline:** The engine that powers batch or real-time predictions to generate actionable insights.

- **Re-training Pipeline:** Keeps your models fresh by updating them as new data or changes come in.

Let's take a closer look at each pipeline to see how they all work together in the ML lifecycle!

#### 1. Initial Training Pipeline

This is where every ML project kicks off. It's all about taking a model from an idea to something ready for action. Here's how it works:

![](/images/blogs/11.ci-cd-pipeline/2.ci-cd-pipeline.jpg)


##### Step 1.1: Getting Your Data Ready

Think of this step as gathering the ingredients to build a car. You need all the parts - wheels, steel, an engine - before you can even think about assembly. For machine learning, it's the same: you need to collect and prepare your input data first.

At my workplace, we use **Google Cloud** for this, relying on tools like **Google Cloud Storage Buckets**, **BigQuery tables**, and **Vertex AI Jupyter Notebooks** to handle everything. Depending on your setup, your raw data will live in something like **Cloud Storage** or **BigQuery**. Once you've got your data, the next step is to clean and process it - just like polishing car parts and making sure everything's in the right shape for assembly. Then it's ready for the real machine learning work in **Vertex AI**.

This step might take time and expertise, but it's the foundation for everything else in your pipeline. Get it right, and you're setting yourself up for success.

##### Step 1.2: Experimenting and Building the Model

![](/images/blogs/11.ci-cd-pipeline/3.ci-cd-pipeline.jpg)

Now that you've got all the parts ready, it's time to start assembling the car. This is where the real work happens - you experiment, test, and refine until everything comes together perfectly.

In machine learning, this is all about figuring out what works. I use **Vertex AI's Jupyter Notebooks** to load the cleaned data and try different tools like PyTorch, Scikit-learn, or TensorFlow. Here's how it usually goes:


- **Feature Engineering**: This step focuses on extracting meaningful patterns from the data; like designing each car part to ensure it works well.

- **Testing Models**: Similar to testing car prototypes, you experiment with different models and fine-tune them to find the best performer.


The great thing about Vertex AI notebooks is how smoothly they integrate with GitHub. You can track everything - your code, parameters, and results - so nothing gets lost along the way. It's like having a detailed log of every tweak and test you've done, making the process organized and efficient (Please Note: Depending on your setup, you might use other cloud-based tools like **Amazon SageMaker** or **Azure ML Notebooks** instead. They are all on par.)

##### Step 1.3: Packaging and Registering the Model

![](/images/blogs/11.ci-cd-pipeline/4.ci-cd-pipeline.jpg)

After your car (or model) is built, the next step is to get it ready for delivery so it can perform anywhere it's needed. Imagine shipping a car from a factory to dealerships around the world. Whether it's going from Thailand to Sydney or China to California, the goal is the same - when it arrives, it's in perfect condition and ready to hit the road. That's exactly what we aim for with your machine learning model.

Here's how it works:

- **Containerization**: Think of this as putting the car into a secure shipping container. Your model is packed into a Docker image, making it portable and ready to go anywhere.

- **Publishing**: Once it's packed, the model is "shipped" by uploading it to Google Cloud Artifact Registry. This makes it versioned and easy to access when needed.

- **Registration**: Finally, just like registering a car before it hits the road, your model is added to Vertex AI's Model Registry. This keeps track of all your versions and ensures smooth deployment.

With these steps done, your model is ready to "hit the road" and perform wherever it's needed!

##### Step 1.4: Automating the Process

![](/images/blogs/11.ci-cd-pipeline/5.ci-cd-pipeline.jpg)

Once your model is packaged, shipped, and registered - just like a car arriving at its destination - it's time to make the whole process more efficient. Imagine if you had to manually handle every step for each car: loading it into the container, tracking the ship, unloading it at the port, and registering it. That would take forever, right?

This is where automation comes in. Instead of doing the same steps over and over, you streamline the process to save time and effort.

Here's how it works:

a) Break down your training pipeline - steps like data prep, model training, and deployment - into smaller pieces. Each component handles a specific task, like tools on an assembly line.

b) Connect these components into one seamless, automated pipeline that takes care of everything from start to finish.

With this setup, you can focus on building and improving your models while letting the pipeline handle all the repetitive, behind-the-scenes work!

#### Step 2: Prediction Pipeline

Once your model is trained, it's time to put it to work - just like a car hitting the road after all the manufacturing is done. The prediction pipeline is where your model starts turning raw data into useful insights, whether through batch processing or real-time predictions.

![](/images/blogs/11.ci-cd-pipeline/6.ci-cd-pipeline.jpg)


##### How Predictions Are Automated

Here's how it works:

a) First, the pipeline pulls the target data from Google Cloud Storage Buckets - think of this as fueling up the car.

b) Next, the model processes the data and saves the predictions back into storage buckets - just like the car delivering goods to their destination.

To make things even easier, you can automate this process with a **Cloud Function**. For example, if your business needs daily predictions, you can set it up so the pipeline automatically runs whenever new data arrives - like scheduling your car for daily deliveries or school drop-offs.

##### How Businesses Use These Predictions

These predictions can drive all kinds of business decisions, such as:

a) Creating **data visualizations** with BI tools to uncover trends.

b) Launching **personalized marketing campaigns** through email, SMS, or web notifications.

c) Running **targeted ads** across platforms like video, display, or search.

It's all about turning your data into action, just like putting your car to work after all the preparation!

### 3. Re-training Pipeline

No car stays in top condition forever - roads change, wear and tear happens, and maintenance is essential to keep it running smoothly. The same applies to machine learning models. No model stays perfect forever - data changes, patterns evolve, and models need updates to stay accurate. That's where the re-training pipeline comes in. It helps your ML systems keep up with the times and deliver their best performance, just like a car that gets regular servicing.

![](/images/blogs/11.ci-cd-pipeline/7.ci-cd-pipeline.jpg)

#### Types of Re-training Pipelines

There are three main ways to re-train your model, depending on what's needed:

a) **Incremental Re-training**: This adds new data to the model to keep it up to date, like topping off the oil or adding fuel to a car. It's great for regular updates, like monthly refreshes.

b) **Full Re-training**: This starts from scratch, using all your data for a fresh rebuild. Think of it as overhauling the car's engine - usually done less often, like quarterly updates.

c) **Drift Correction**: This kicks in when something goes wrong, like a car repair after an unexpected breakdown. It's used to fix a drop in the model's performance.

#### How the Pipeline Works

![](/images/blogs/11.ci-cd-pipeline/8.ci-cd-pipeline.jpg)

Every re-training pipeline has three main steps:

a) **Data Prep**: This is where you get the updated dataset ready for training - like gathering the tools and parts needed to service a car.

b) **Re-training**: Think of this as servicing the model. It could be a quick update or a complete rebuild, depending on what's needed.

c) **Model Registry**: Just like keeping a car's maintenance records, the updated model is saved in the registry to track its latest version.

Once the re-trained model is ready, it connects back to the prediction pipeline. This creates a smooth, ongoing cycle where the model keeps improving and stays in sync with new data over time.


### Wrapping Up

For data scientists, analysts, or BI professionals, understanding how to build and manage these pipelines can be a game-changer. It's not just about theory - these skills help you create scalable ML systems that really make an impact.

I hope this gives you a simple and practical look at how CI/CD pipelines work, especially in marketing and analytics. In future posts, I'll dive deeper and show you how to build a complete, end-to-end pipeline using open-source datasets. These lessons have been invaluable in my career, and I hope they'll help you too!

































Let me share my perspective as someone who works in the **marketing analytics spectrum**, where structured data and repeatable workflows rule the day.

#### Why Agentic AI Sounds Tempting

There's no doubt that agentic AI has its place in marketing analytics:

a)  Amazing for [digging into customer reviews, product feedback, and online ratings](https://srees.org/project/review-nlp) using methods like **topic modeling, text clustering, and sentiment analysis**.

b) Works beautifully to create **automated insights** and **Q&A bots** from processed data, like **AI commentary** on BI dashboards.

c) Perfect for **brand monitoring** via social media feeds or **audience text mining** through chatbot conversational feeds.

The vision is compelling: let AI take care of the grunt work while we focus on strategic decisions. But while agentic AI shines in certain areas, it's not always the right tool for every job - especially when it comes to structured data and proven models.


- **What KPIs will we track?**
    a) Retention rates for dormant and inactive customers.
    b) Average purchase value for high spenders.
    c) Engagement metrics for potentially valuable customers.
- **How will we monitor these KPIs?**
    a) Build a framework to track performance by segment and ROI from tailored campaigns directly through your CDP.
    
    
#### Structured Data and Proven Models: If It Ain't Broke, Why Fix It?

Take a typical example in marketing analytics: predictive analytics and forecasting.

In my work, I've built [domain-specific forecasting pipelines using models like ARIMA, Facebook Prophet, or Holt-Winters](https://srees.org/project/predict-sales). These scripts are fine-tuned to the white goods retail industry, running seamlessly on cloud platforms to generate daily forecasts. They're **explainable, transparent, and customized** to our unique needs.

> "This segmentation identifies 30% of our customers as dormant, representing untapped revenue opportunities of $X"
> 


Now, someone might say: "Why not replace all that with agentic AI?" Here's my response: Why would I?

When my current system already:

a) Explains its predictions clearly (something stakeholders love),

b) Runs computationally fast without blowing up cloud costs,

c) Gives me full control to tweak or debug the process, and

d) Reliably delivers actionable forecasts, there's no real benefit to wrapping it all in agentic AI just for the sake of it.


#### Why Simplicity Works Every Single Time

Let's look at another example: **predictive customer lifetime value (CLV)**.

In another case, I use [cohort retention matrices and supervised regression models to predict each customer's lifetime value](https://srees.org/project/predict-cltv) based on their evolving purchase behavior. It works beautifully, delivering granular insights into why certain customers are high- or low-value. Stakeholders love it because the underlying models are industry-proven, stood the test of time, straightforward and interpretable. They provide well-defined structures and outcomes making it easier for stakeholders to understand and trust decisions.


Here's where agentic AI might add value:

a) Generating automated insights for internal stakeholders:"High-CLV customers are 40% more likely to purchase premium add-ons. Target them with upsell offers!"

b) Dynamically integrating new data sources, like unstructured social media engagement data.

But replacing the core model with agentic AI? Not necessary. The system already works, and introducing AI agents doesn't magically make it better.

#### The Value of Agentic AI: Where It Shines

That said, agentic AI isn't without merit. It can complement existing systems in specific ways, like:

a) **Real-time Adaptability:** For industries where demand can change unpredictably, AI agents with reinforcement learning might adapt faster than traditional models.

b) **Proactive Recommendations:** Beyond forecasting, agentic AI could autonomously suggest actionable steps, like reallocating inventory or adjusting pricing based on predictions.

c) **Automating Insight Generation:** For example, adding an AI commentary layer that translates raw forecasts into stakeholder-friendly narratives:"Sales for Product A are projected to increase by 15% next month, driven by a seasonal uptick and a strong promotional push."


These are valuable enhancements - but they don't necessarily require overhauling an already well-functioning pipeline.


#### Practical Tips: Use AI Where It Matters

![](/images/blogs/10.limits-agentic-ai/2.limits-of-agentic-ai.png)


To all the budding marketing analytics professionals and graduates out there wondering which road to take in this ever-evolving field, here are some of my personal tips - drawn from years of learnings, failures, and experiments with classic ML models and LLMs.

a) **If It's Working, Don't Reinvent the Wheel:** The old adage still applies: Don't fix what isn't broken! If you've already built a system that is **customized to your domain, highly transparent, and tailored to your business logic**, stick with it! Replacing these systems with a black-box AI model might bring marginally different results, but at the cost of explainability, control, and significant **cloud costs.**

b) **For Structured Data, Keep It Classic:** Classical machine learning models like **XGBoost, logistic regression, or random forests** are highly effective for structured/tabular data. They're computationally efficient, interpretable, and easy to debug - perfect for solving core business problems.

c) **Leverage Agentic AI Where It Shines:** For tasks involving unstructured data, real-time adaptability, or generative automation, agentic AI and LLMs are game-changers. Automating repetitive tasks or enriching workflows with these tools can unlock significant value. But when such agentic AI systems are put into production, please always keep in mind that there will be trade-offs in control, explainability, and customization.

d) **Focus on the Problem, Not the Tool:** After all, AI is a tool - its value lies in how and where you apply it. Always ask: Does this AI solution genuinely solve a problem? Avoid falling into the trap of overengineering or using AI for its own sake.

### Closing Thoughts

As a marketing data science and analytics professional, we're often at the crossroads of innovation and practicality. Our job is not just to build models but to bridge the gap between technical systems and business stakeholders. This means explaining the "why" behind results and ensuring that our solutions are aligned with business goals. One of the biggest challenges with agentic AI is its lack of transparency. When stakeholders need to justify decisions - especially those tied to real money - it's essential to provide clear explanations for the model's outputs.

Think about it from their perspective - stakeholders are investing real money into campaigns and customers your model selects. When everything works well, it's great. But in the real world, things do go wrong, and when they do, you need to be able to walk them through what happened, why it happened, and how you'll fix it next time. That's what truly earns trust. Like in life, in most cases, simplicity and transparency are what drive real results and lasting business relationships.

So, the next time when someone suggests that agentic AI will "transform everything," ask yourself: does it genuinely solve a problem? Or is it just a new coat of paint on a system that's already doing the job?

In short, don't let the hype steer you away from what already works beautifully. AI can enhance our systems, but it should never replace the logic and workflows we've meticulously refined through years of experience - processes that have consistently stood the test of time.