# *Unlocking Restaurant Revenue: Analyzing Key Strategies for Boosting Profitability*


<img src="img/cover-main.webp" alt="strategy" width="450">


- [Background](#background)
- [Data exploration and insights discovery](#Data-exploration-and-insights-discovery)
   + [notebook](exploring-data.ipynb)


### *`Background`*


*The Power of Smart Strategies and Data-Driven Decisions*

In today's highly competitive restaurant industry, maximizing revenue is more crucial than ever. Restaurants need to adopt innovative strategies and make data-driven decisions to stay ahead. 

In this analysis, we explore how smart decision-making, strategic planning, and in-depth analysis can significantly boost restaurant revenue. Leveraging advanced machine learning algorithms and causal inference methods, we can uncover hidden patterns and insights that drive profitability. 

By understanding the importance of effective revenue management, we can implement targeted actions that not only enhance the customer experience but also optimize operational efficiency. Join us as we delve into the transformative potential of combining cutting-edge technology with strategic foresight to elevate your restaurant's financial performance.


### *Data exploration and insights discovery*


To begin implementing the strategy, we first need to delve into the data to gain a better understanding of market behavior. This detailed analysis will provide the insights necessary to tailor our approach effectively.

In the following box chart, it's evident that there are no atypical values. However, what's striking is the minimal expenditure on marketing, coupled with the observation that certain restaurants have a notably low customer count.


<img src="img/boxplot_price_customers.png" alt="boxplot-price-customers" width="1000">


In examining the box charts, we find no atypical data in key metrics such as menu prices, marketing spend, average customer spending, and the number of customers. However, a closer look reveals some outliers in monthly revenue, with notably high values that appear somewhat abnormal. To gain a clearer understanding of this anomaly, further exploration of the data is necessary. 

Additionally, the presence of negative revenue figures raises concerns, suggesting instances where costs surpass income—a scenario that warrants investigation. One plausible hypothesis is that the restaurant is new and thus invests heavily in marketing, anticipating a surge in clientele.

Turning to specific features:

Menu prices predominantly fall within the range of 20 to 40 dollars, though the influence of cuisine type remains undetermined.
Marketing expenditure appears disproportionately low compared to menu prices, with no restaurant exceeding a 20-dollar spend.
Average customer spending mirrors menu prices without notable anomalies, alongside a stable customer count. However, deeper analysis is required to assess the impact of promotions on these metrics and ascertain whether promotions correlate with revenue growth or increased customer visits.


<img src="img/boxplot_monthly_revenue.png" alt="boxplot-monthly-revenue" width="1000">


Now, considering our initial exploration into the relationship among income, revenue, and pricing, it becomes evident that a positive correlation exists. In other words, when there's an uptick in marketing expenditure and menu prices, revenue tends to rise accordingly.

While this association isn't overwhelmingly strong, it provides valuable insights. Firstly, we discern that menu pricing doesn't significantly sway sales. Instead, our focus shifts to analyzing customer numbers and average spending per customer. By first determining **elasticity** and subsequently assessing the influence of the average transaction value on revenue, we gain a deeper understanding of their impacts.


<img src="img/corr_heatmap.png" alt="correlation-heatmap" width="750">


`marketing efficency`

Moving forward, a crucial step involves calculating the marketing ROI. A high ROI indicates that you're generating substantial income for every dollar invested in marketing, showcasing the efficiency of your marketing expenditure. On the other hand, a low ROI suggests the need to refine your strategy and optimize your marketing spend to enhance efficiency and maximize returns.

As we analyze the revenue data, we notice some outliers that can obscure the overall understanding of ROI across all restaurants. To gain deeper insights into the business, we will conduct further analysis to navigate beyond these anomalies.


<img src="img/boxplot_roi_marketing.png" alt="boxplot-roi-marketing" width="1000">


As depicted in the chart below, Japanese cuisine stands out with the highest revenue. This suggests that perhaps the average menu price for Japanese dishes is also the highest among all cuisine types. Moreover, it's plausible that this particular cuisine type is responsible for the outlier points in the ROI analysis.


<img src="img/roi_cuisine.png" alt="roi-cuisine" width="1500">


<img src="img/avg_spend_cuisine.png" alt="avg-spend-cuisine" width="800">


<img src="img/cuisine-menu-price.png" alt="cuisine-menu-price" width="800">


The reality diverges significantly from our earlier assumption that Japanese cuisine commands a higher average menu price. Upon closer examination, it becomes evident that the average customer spending across all cuisine types remains fairly consistent. Surprisingly, the menu prices for Japanese dishes are comparatively lower than those for Mexican cuisine, which appears to be priced at a premium. This unexpected finding prompts a deeper investigation to understand the underlying reasons for this particular behavior.


`promotion impact`

In our quest to understand the impact of promotions on customer visits, the violin chart provides valuable insights. Comparing the median points, we observe that when promotions are applied, there's a notable increase in customer numbers compared to periods without promotions. Additionally, the dispersion of customers appears to be more concentrated during promotional periods. This suggests a potential link between promotions and customer turnout, although it's premature to establish causality at this stage. It's important to note that other external factors not captured in the dataset, such as day of the week, holidays, and weather conditions, may also influence customer traffic and warrant consideration in our analysis.


<img src="img/violin_customers.png" alt="violin-customers" width="1000">


`price impact`

The chart below presents a nuanced view of Japanese cuisine within the market. While it does not feature the most expensive menu or the highest average customer spending, it boasts the highest marketing ROI. However, it's important to note that there is an outlier in the data, urging caution in our conclusions. To formulate the best strategy moving forward, we must delve deeper into the data to understand the preferences and behaviors of different customer segments. Additionally, a comprehensive overview that includes revenue metrics will be essential in aligning our marketing efforts with the broader business objectives. This thorough analysis will ensure that our strategies are both informed and effective.


<img src="img/avg_spend_and_price_cuisine.png" alt="avg-spend-price-cuisine" width="1000">


The latest chart offers valuable insights into the visitor counts at our establishments, with Japanese food restaurants seeing a notable increase in patronage. This trend may be influenced by a variety of factors, such as prime locations, compelling promotions, or substantial marketing investments. Additionally, there are indications that **pricing strategies** are also impacting customer attendance. Understanding these dynamics is crucial for optimizing our approach to attract and retain more customers efficiently.


<img src="img/avg_spend_and_price_and_visitors_cuisine.png" alt="avg-spend-price-visitors-cuisine" width="850">


`relation among the revenue, the number of visitors and reviews`

The radar chart provides some intriguing insights into our culinary offerings. Consistent with previous data, Japanese cuisine generates the highest average monthly revenue, although the figures are quite similar across all types of cuisines. Interestingly, Mexican food, despite not garnering a large number of reviews, maintains a robust visitor count. On the other hand, Italian cuisine, which has slightly more reviews than Mexican, surprisingly sees fewer customers, even with comparable menu prices. This suggests that other variables such as promotions, location, or service quality might be influencing these trends. Currently, we can only speculate whether promotions directly affect customer turnout. In the upcoming sections, we will engage in causal inference analysis to determine the impact of these factors more definitively.


<img src="img/radar_chart.png" alt="radar-chart" width="1700">

<!-- #region -->
### `insights`

*Distribution Data Analysis:*
- The charts reveal mostly typical values across various features with notable exceptions in the revenue data. Japanese cuisine displays extreme revenue values, including some restaurants experiencing negative revenue, indicating financial losses. Conversely, most menu prices and average customer spending lie within the \\$20 to \\$40 range, with minimal marketing expenditure across all establishments.

*Revenue Correlation Insights:*
- Analysis indicates a positive correlation between revenue and both marketing spend and menu price. This suggests that higher menu prices and increased marketing investments may lead to an uptick in revenue, although the correlations are modest and not necessarily indicative of causation.

*Marketing ROI and Cuisine Type:*
- Japanese cuisine consistently shows the highest marketing ROI, possibly skewed by outliers. This suggests that Japanese restaurants may be more effective in managing marketing efforts like promotions. However, this does not imply causality. Notably, the Japanese menu is not the priciest, nor does it attract the highest spending per customer.

*Impact of Promotions:*
- Violin chart analysis demonstrates that promotions significantly impact customer numbers, with a noticeable uptick in visitors during promotional periods. This could suggest a market responsive to price changes (elastic market), although it remains speculative without more detailed information.

*Price Versus Customer Numbers:*
- There appears to be no direct correlation between menu prices and customer volume. For instance, despite having lower menu prices, Italian cuisine does not attract more customers compared to American cuisine, which has higher prices and more visitors. Location may play a role in this dynamic, but this remains an assumption.

*Radar Chart and Customer Insights:*
- The radar chart highlights an interesting trend: despite Italian cuisine receiving slightly more reviews than Mexican cuisine, the latter maintains a higher customer count, possibly due to effective promotional strategies.


**Current Insights:**
- Reviews do not necessarily translate to increased visitor numbers.
- Japanese cuisine sites exhibit the best performance in terms of ROI.
- Menu pricing appears to have a minimal impact on demand, suggesting market inelasticity.
- Promotions significantly influence customer numbers, enhancing restaurant visitation despite low marketing expenditure.

These observations underscore the complex interplay of factors affecting restaurant performance and customer behavior.
<!-- #endregion -->

```python

```

```python

```

```python

```

```python

```
