---
title: 'How good is Google Looker?'
date: 2020-12-10 00:00:00
featured_image: '/images/blogs/4.google-looker/blog_addin1.jpg'
excerpt: Key points to evaluate Google's Looker vs. Tableau/Power BI/Qlik
---


<style>
body {
text-align: justify}
</style>

### "Which Business Intelligence (BI) tool is better for my business"?

With so many key players in the Business Intelligence and Visual Analytics market, this is one of the most important and frequent questions that need to be addressed in any techno-functional department of an organization.

Google is getting deeper into enterprise software with its latest acquisition of Looker for $2.6 billion. I just had a quick look into Looker and I think its definitely worth trialling out for few weeks. Most of my experience with BI tools fundamentally revolves around the traditional ones in the market like Power BI, Tableau, Qlik, Omniscope (and few other R/Python scripted ones like Plotly and Shiny too). Sharing some of my quick thoughts here while looking at the features of Looker in case if it benefits anyone during its product evaluation phase:

#### 1. Big Query & Looker: 

If your organization have an increasingly large volume of data hosted in Google Big Query, using Looker as an online BI platform might be of an advantage there. But then Google Data Studio fairly does that job well. So we might have to check with the folks at Google Cloud to see how Looker would be better positioned in this space (with regards to pricing and performance).

#### 2. Fully Online:

Unlike Tableau, Power BI or Qlik, Looker seem to be a fully online SaaS model. It would be interesting to see how they handle multiple versions of the files while building large scale BI products. Many a time cost optimization happens when we sample a portion of our dataset from the cloud/ on-prem into the local host to build all the necessary elements of a dashboard before pushing it back to cloud for end-user consumption. So, if it's fully online, would Looker allow us to create workspaces - like say for Finance, Marketing, HR etc. and are there any storage limits involved?

#### 3. Integration with R, Python & SQL: 

One of the paradigm shifts in the BI industry over the last few years is their flexibility to incorporate R/Python scripted widgets (besides the traditional ones like SQL that gets pointed to the databases). Integrating R and Python helps a lot with data science projects considering that you get a choice to code and build any visualization widgets in case if it doesn't exist in their BI suite. Also, you can actually paste your R/Python script in the backend data model layer; enabling data science to co-exist with Business Intelligence Platforms in near real-time. Does Looker have integration with R & Python?

#### 4. No. of Third Party API End Points: 

From a marketing standpoint, this is a huge plus if we rely on traditional BI players like Tableau/Power BI/Qlik as they have innumerous API endpoints (ranging all the way from CRM's, cloud databases & storages, third party ad-exchange servers, the independent site served ad buying platforms like Facebook, LinkedIn, Twitter etc.). Does Looker offer a similar list of third party API endpoints (besides the Google suite of Google Ads, DoubleClick etc.)?

#### 5. Pricing: 

Looker hasn't put up pricing on their website. But then little research shows that they charge on average $3000 to $5000 USD per month for 10 users and then $50 per month for each additional user. It sounds on par with the pricing of Tableau but then Tableau has a better pricing strategy for developer vs. end-user particularly when we purchase 'Tableau Online' I guess.

#### 6. Community: 

Not sure if Looker has a community version like Tableau/ Qlik. Community Reporting Galleries with preset templates and user base helps us a lot to adopt any BI tools quickly and efficiently. It simply shows other BI enthusiasts work on public data sources and gives us a platform to resolve any queries instantly (like a vendor platform of 'StackOverflow')

#### 7. Reporting Automation: 

In order to automate the dashboards in real-time, Power BI has on-premise/ enterprise gateways whilst Tableau does it through their Servers. Hence it would be interesting to see the offering from Looker on that space. particularly when we have a mix of on-premise and cloud data sources.

Overall I felt that Google's Looker has a lot of visual analytics capabilities and data discovery use cases, and it also opens up new avenues like collaborative data sharing etc. Unlike Power BI or Tableau, Looker uses a complete semantic model for storing all the business logic providing a single source of truth for the enterprise. Hope some of these checkpoints help you next time if you have been given the task to evaluate Google's Looker as a prospective BI platform for your department or organization.

Thank you
