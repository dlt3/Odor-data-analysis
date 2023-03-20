# Odor data analysis
This study focus on develop a odor predict model and interpret the model's classification result by using explainable AI method.

#### Reference
- https://doi.org/10.3390/app12062826
- https://doi.org/10.3390/app122412943

### Research purpose
- Prevention of odor in pig barns by managing chemical substances (odor substances) that affect odor generation
- Creation of an optimal prediction model for complex odors using 15 odorous substances
- Identification of the influence of odorous substances on complex odors and the interaction effect between odorous substances
- Creation of a complex odor classification prediction model using 15 odorous substances and measurement-related variables
- Prevention of bad smell in pig houses by managing chemical substances (odor substances) that affect odor generation

### Data information
- explanatory variable : Complex odor
- response variable : 15 odorous substances
    - Ammonia
    - Sulfur compounds: Hydrogen Sulfide, Methyl mercaptan, Dimethyl sulfide, Dimethyl disulfide
    - Volatile Organic compounds: Acetic acid , Propionic acid, Butyric acid, Iso-Butyric acid, Valeric acid, Iso-Valeric aic, Phenol, para-Cresol, Indole, Skatole


![image](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F234ff89f-fcdf-4385-b1d8-e56757189364%2FUntitled.png?id=799b5c0b-80ed-4f8c-a7ef-fdd8ddaec4fe&table=block&spaceId=6cc23a96-8110-4f80-9a0b-4eb515095500&width=2000&userId=e639e6c1-7dd8-4d51-97de-be9ead475dc3&cache=v2
)

### Analysis process

#### Research 1
- Compare different analysis processes to find the optimal predictive model
- Data problems and solutions
     - High missing rate: Considering the fact that the missing rate may be high considering data collection through sensors in the future, consider the replacement  method rather than the missing value removal method
     - Small amount of data: Model validation through the Leave-One-Out Cross Validation (LOOCV) method that can be used when there is little data
- Data pre-processing
     - Missing imputation: Simple imputation (mean, median), Multivariate imputation (bayesian), Multiple imputation (bayesian ridge, gaussian process regression, KNN)
     - Feature preprocessing: standardization, Partial Least Square (PLS), Principal Component Analysis (PCA)
- Prediction models: Regression, SVM, RandomForest, ExtraTree, XGBoost, DNN
- Model Verification: Using R-square, MAPE through LOOCV
- Additional Analysis: Correlation Analysis, Principal Component Analysis(PCA), Identification of predictor feature importance

![image](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0d89114d-efcd-4735-98a4-61ec2deece1b%2FUntitled.png?id=e90a3b0d-fe84-4106-8f77-124a8a2adc9e&table=block&spaceId=6cc23a96-8110-4f80-9a0b-4eb515095500&width=2000&userId=e639e6c1-7dd8-4d51-97de-be9ead475dc3&cache=v2)

#### Research 2
- Features related to measurement: measurement time (year, month, day), measurement location (inside the pig barn, outside the pig barn, site boundary)
- summary
     - Perform data preprocessing based on primary research and compare multiple machine learning models
     - Minimize overfitting by analyzing 30 times and select the optimal model through 8 evaluation indicators
     - Identification of the influence and interaction effect of odor spray through the XAI method
- Data pre-processing
     - Complex odor: Conversion of continuous data into binary classification data in the form of emission possible / non emission in accordance with the domestic odor prevention law
     - Measurement-related variables: Measurement time variables are converted into seasonal variables, followed by One-Hot Encoding, and measurement location variables One-Hot Encoding
     - Variable preprocessing: Multivariate imputation (bayesian ridge) & Standardization
- Prediction models: k-Nearest Neighbor, SVC, RandomForest, LightGBM, ExtraTree, XGBoost
- Model validation: F1-score, Accuracy, Sensitivity, Specitiv
- Identification of influence: XAI - Partial Dependence Plot, variable importance
- Additional analysis: correlation analysis and VIF (continuous variable), ANOVA (categorical variable)

![image](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F59cd098c-2ae7-4cd7-883f-aebe8842bdee%2FUntitled.png?id=6721a0d4-3f73-4d07-b4ac-ccf79205a479&table=block&spaceId=6cc23a96-8110-4f80-9a0b-4eb515095500&width=2000&userId=e639e6c1-7dd8-4d51-97de-be9ead475dc3&cache=v2)

