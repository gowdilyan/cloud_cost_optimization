# Cloud Cost Optimization and Expected Spend Forecasting

**Author**

Gowdilyan Subramaniam

#### Executive summary

Most teams only see their cloud bill once a month, so if spend goes up they usually find out weeks later. For this project I used one year (2023) of daily cloud billing data to see if past spend can be used to forecast next month's cost, and if that forecast can warn us early when spend goes higher than normal.

In the EDA I found two main patterns. Weekends are about 24% cheaper than weekdays, and instead of growing slowly, spend jumps up or down in steps (up 16% in June, down 10% in August and up 29% in November).

I compared six models from the course against two simple baselines. My final model, SARIMA, was off by about 5.7% per day in normal months, which is under my goal of 10%, and its December total was only 1.9% off. None of the models could predict the jumps, since there's nothing in past spend that shows a jump is coming. But when I used the forecast's 95% range as an alert, it flagged the June and November jumps on the first day and didn't flag anything in the normal months I checked.

Based on this, I'd recommend using the forecast for monthly budgeting, setting up a daily alert based on the forecast range, focusing cost reviews on prod (77% of the spend), and scaling down dev and staging on weekends.

#### Rationale

Cloud costs change every day, and a new workload or a test environment someone forgot to turn off can add a lot to the bill before anyone notices. Finance needs a good estimate for next month's budget, and engineering needs to know early when spend starts going up so they can look into it. A forecast of the expected spend can help both.

#### Research Question

How can an organization use historical time series analysis to build an accurate forecasting model for its expected cloud infrastructure costs?

#### Data Sources

I used the [Cloud Budget Dataset on Kaggle](https://www.kaggle.com/datasets/rasikaekanayakadevlk/cloud-budget-dataset), specifically the main file `cloud_budget_2023_dataset.csv` (in the `data` folder). It has 54,750 rows and 40 columns, with 150 billing line items per day for all of 2023 across AWS, Azure and GCP. Total spend for the year is about $354,000, or about $29,500 a month. My target is `amortized_cost`, which is the cost after discounts, summed per day.

Changes since my Module 16 proposal:
- The proposal mentioned CPU, RAM and storage utilization as predictors, but the dataset doesn't have those columns. The only usage metric is `usage_quantity`.
- I expected daily and weekly cycles. The data is daily, so there's no within-day cycle, and only the weekly season turned out to matter.
- I also expected higher spend at the end of the month. I tested this and didn't find it.
- I expected holiday trends too. Weekends are clearly cheaper, but holidays aren't. The holiday Mondays on long weekends cost the same as normal Mondays, and Thanksgiving and Christmas cost about the same as or more than a normal weekday. With one year of data each holiday only happens once, so I couldn't test this properly, but it looks like the drop only follows Saturday and Sunday on the calendar, not days off.
- The budget and anomaly columns weren't useful (budget status is always "under" and no anomalies are flagged), so I compare the forecast to actual spend instead of the budget.
- In the proposal I planned to use MSTL and SARIMA. In the end I compared six models plus two baselines, and tested them on nine months instead of one.
- The idea of flagging spend above the forecast turned into the alert described in the results.
- The dataset is synthetic.

#### Methodology

In the first notebook I cleaned the data (checked missing values, duplicates and missing days, and dropped columns that had only one value) and looked at outliers with the IQR rule. The outliers were real prod costs, and the high days were part of the November/December increase, so I kept them. For the EDA I looked at spend by category, the distributions, usage vs cost, correlations, cost over time and the weekly pattern. I also checked the time series itself with an MSTL decomposition, the ADF test for stationarity and ACF/PACF plots. Finally, I used t-tests to check whether the patterns I saw were real: weekday vs weekend, the weekend drop in each environment, October vs November, and the end of the month vs the rest of the month.

In the second notebook I built the models. The two baselines are very simple: one repeats the last day and the other repeats the last week. From Module 10 I used three time series models: ARIMA, SARIMA (ARIMA with a weekly season) and STL + ARIMA (which removes the weekly pattern, forecasts the rest with ARIMA and adds the pattern back). For each one I tried a range of settings and picked the one with the lowest AIC. I also tried three regression models using day of week, month and the cost from 5 and 6 weeks earlier as features: Ridge regression, Random Forest and Gradient Boosting. I tuned these with `GridSearchCV`, using `TimeSeriesSplit` so the model is always trained on earlier days and checked on later ones.

To evaluate the models, I first trained on January to November and tested on December. Since one month could just be lucky, I then repeated the whole process (including the tuning) for every month from April to December, each time only using the data before that month.

My main metric is MAPE (mean absolute percentage error), which is the average percent the forecast is off by each day. I picked it because it's easy to explain to anyone and it doesn't depend on how big the bill is. MAPE can be a problem when actual values are close to zero, but the lowest daily cost here is about $600, so that isn't an issue. I also report MAE in dollars and the error on the monthly total.

#### Results

Prod makes up about 77% of the spend, staging 17% and dev 6%. Provider, region and business unit are split almost evenly.

![Spend by category](images/spend_by_category.png)

Spend doesn't grow smoothly. It went up 16% in June, came back down 10% in August and jumped 29% in November. A t-test shows the November jump is real and not just noise (p < 0.001).

![Daily cost trend](images/daily_cost_trend.png)

Weekends cost about $246 a day less than weekdays ($796 vs $1,042), and this difference is also significant (p < 0.001). Dev only drops about 20% on weekends, compared to about 24% for staging and prod, which is a bit surprising since dev is the environment you'd expect to be idle on a weekend. I didn't find any end-of-month effect (p = 0.59).

![Weekly pattern](images/weekly_pattern.png)

For the first test, I trained every model on January to November and forecast December:

| Model | MAPE (daily error) | MAE | Error on the December total |
|---|---|---|---|
| STL + ARIMA | 5.6% | $66 | -1.4% |
| SARIMA | 6.4% | $71 | +1.9% |
| Seasonal naive | 7.3% | $87 | -1.5% |
| Gradient Boosting | 7.3% | $89 | -5.2% |
| Random Forest | 8.0% | $88 | -4.9% |
| Ridge | 12.5% | $152 | -13.1% |
| ARIMA | 12.5% | $137 | -0.4% |
| Naive | 13.9% | $141 | +7.6% |

![December forecast](images/december_forecast.png)

Then I repeated the test for each month from April to December. In June, August and November the spend level shifted, and every model missed by 10% to 26% in those months. In the other six months the best models were off by about 6% or less.

![Backtest](images/backtest.png)

| Model | Normal months (Apr, May, Jul, Sep, Oct, Dec) | Shift months (Jun, Aug, Nov) | All 9 months |
|---|---|---|---|
| SARIMA | 5.7% | 17.4% | 9.6% |
| Random Forest | 6.0% | 15.6% | 9.2% |
| STL + ARIMA | 6.0% | 17.0% | 9.7% |
| Gradient Boosting | 6.3% | 14.6% | 9.1% |
| Seasonal naive | 7.1% | 16.9% | 10.4% |
| Ridge | 7.8% | 16.0% | 10.6% |
| ARIMA | 11.2% | 18.1% | 13.5% |
| Naive | 15.0% | 20.2% | 16.7% |

The weekly pattern turned out to be the most important thing for the models to capture. ARIMA, which doesn't have a weekly season, was barely better than the naive forecast. The top four models were within less than one percentage point of each other in normal months, so I don't think the small differences mean much. The tree models did a bit better in the shift months but were less consistent in normal months (in December they were about 5% too low). I think with only one year of data, the regression models just didn't have enough history to learn more than the time series models.

I chose SARIMA as my final model. It had the lowest error in normal months and was never worse than 6.4% in any of them, its residuals look like random noise, and it gives a 95% range for each day, which I used for the alert below.

Since no model could predict the jumps, I wanted to see if the forecast could at least catch them quickly. I trained SARIMA on the data before each month and counted the days where actual spend went above the 95% range. In November, 24 of the 30 days were above the range, starting on November 1. In June it was 16 of 30 days, also starting on the first day. In October and December, which were normal months, no days went above the range.

![Spend alert](images/spend_alert.png)

#### Recommendations

1. Use the forecast to set next month's budget, and refit it every month with the latest data. It only takes a few seconds to run.
2. Set up a daily alert for when spend goes above the forecast range. If that happens two days in a row, check with the teams what changed (a new launch, a migration, a new workload) and update the budget then, instead of waiting for the bill. In November this would have given almost a month of warning.
3. Focus cost reviews on prod. Saving 5% in prod is worth about $13,700 a year, which is more than what dev costs on all the weekends combined ($5,300).
4. Scale down dev and staging on weekends. This could save up to about $19,000 a year, or about 5% of the total. That's the maximum though, since some non-prod work probably does need to run on weekends.
5. There's no need to plan for higher spend at the end of the month, since the data doesn't show it.

Since the dataset is synthetic and only covers one year, these numbers should be checked on real billing data first.

#### Next steps

- Try the same approach on real billing data with more than one year of history, so the models can learn yearly patterns like the November increase.
- Add known events like launches or migrations as features, so planned jumps can be forecast instead of only caught after they happen.
- Forecast prod, staging and dev separately. Each one has a different weekend pattern, and separate alerts would show where a change came from.
- Try different alert ranges (for example 80% instead of 95%) and see how many real changes and false alarms each one gives.

#### Outline of project

- [Part 1: Data cleaning, EDA and inferential statistics](https://github.com/gowdilyan/cloud_cost_optimization/blob/main/1_cloud_cost_eda.ipynb)
- [Part 2: Modeling, evaluation, spend alert and recommendations](https://github.com/gowdilyan/cloud_cost_optimization/blob/main/2_cloud_cost_modeling.ipynb)

Repository structure:

```
├── data
│   └── cloud_budget_2023_dataset.csv
├── images
├── 1_cloud_cost_eda.ipynb
├── 2_cloud_cost_modeling.ipynb
├── requirements.txt
└── README.md
```

To run the project, clone the repository, install the packages with `pip install -r requirements.txt`, and run the two notebooks in order. Each notebook saves its charts to the `images` folder. The second notebook takes about 3 minutes, mostly because of the nine month test.

Packages used: pandas, numpy, matplotlib, seaborn, scipy, statsmodels and scikit-learn.

##### Contact and Further Information

GitHub: [gowdilyan](https://github.com/gowdilyan)
