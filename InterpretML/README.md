## 使用microsoft的模型可解释框架官方文档为：

https://github.com/interpretml/interpret

## 支持的模型可解释算法：

| Interpretability Technique                                   | Type               |
| ------------------------------------------------------------ | ------------------ |
| [Explainable Boosting](https://interpret.ml/docs/ebm.html)   | glassbox model     |
| [Decision Tree](https://interpret.ml/docs/dt.html)           | glassbox model     |
| [Decision Rule List](https://interpret.ml/docs/dr.html)      | glassbox model     |
| [Linear/Logistic Regression](https://interpret.ml/docs/lr.html) | glassbox model     |
| [SHAP Kernel Explainer](https://interpret.ml/docs/shap.html) | blackbox explainer |
| [LIME](https://interpret.ml/docs/lime.html)                  | blackbox explainer |
| [Morris Sensitivity Analysis](https://interpret.ml/docs/msa.html) | blackbox explainer |
| [Partial Dependence](https://interpret.ml/docs/pdp.html)     | blackbox explainer |

## Setup:

```
pip install interpret
pip install tensorflow
pip install xgboost
```

