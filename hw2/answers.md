### 2 Unbiased Learning-to-rank
Thorsten et al. [1] propose an unbiased learning-to-rank method, known as IPS, that can address inherent bias in user interactions such as clicks.
 
> a. According to their experimental results, How successful is IPS in addressing bias in click data? In the presence of high degrees of bias, how the performance of their model could be improved? (15.0p)
 
The empirical results in the paper show that IPS successfully addresses bias in click data. The propensity SVM approximates the performance of a full-information SVM-Rank with increasing amounts of generated click data. On the other hand, the naiveSVM’s performance does not increase with more data, as its error is dominated by the effect of failing to account for positional bias in clicks. Its performance is flat with respect to the amount of data because more data only helps to decrease variance and not the constant bias of the naive model.
Nevertheless, the performance of the propensity SVM still does suffer from increasing amounts of presentation bias. However, this can be mitigated by getting more training data, as seen in Figure 2 of the paper. The robustness to increasing bias offered by greater amounts of training data is only seen in the propensity SVM, as the naive model cannot take advantage of this effect.
 
> b. One of the implicit biases that are ignored as a result of their IPS formulation is the bias caused by implicitly treating non-clicked items as not relevant. Discuss when this implicit bias is problematic? (25.0p)
 
Assuming no click noise, if an item is not clicked then the item must either not be relevant or not be observed. There therefore is a case where an item is not observed but relevant. This is a problem because the IPS loss formulation in Joachims et al. is weighed by relevance label (which is equivalent to click), therefore an unobserved but relevant item will be misclassified as not relevant because not clicked, and yet not contribute to the loss to be optimized (because its contribution will be multiplied by zero). In cases with really large position bias, it may often happen that a relevant item is not observed. While IPS will help correct the bias of clicked documents, reweighing the contribution of correct clicks by how unlikely they are to be observed, it does not do that for unclicked documents. This will lead to a systematic underestimation of the loss, leading to an imperfect optimization.
 
If there is click noise, then there is a situation in which an item can be not clicked and both relevant and observed. This does not bias the IPS metric, as Joachims et al. show that given enough data the performance of the Propensity SVM trained on a large amount of data is still very robust to increasing click noise (Figure 3).  More generally, they show rigorously that in expectation this click noise is order-preserving for the IPS-driven ranking method. 
 

> c. Propose a simple method to correct for the implicit bias of non-clicks. (20.0p)
 
A simple way to correct for the bias of these non-clicks is to find a method through which at least some relevant but not observed items are observed (and by assumption then clicked and therefore can contribute to the loss). A first approach could be to randomize the order of all the retrieved documents, but that would make for a very ineffective information retrieval system. Instead, at every training instance you could select $m > 0$ pairs of indeces of the retrieved ranking and switch the order of the documents. This would enable some documents that would not be observed to be observed, thereby contributing to the loss and mitigating this bias. Nevertheless, this is not a robust way to completely remove the 



### 3 LTR with IPS
 
> a. Explain the LTR loss function in Thorsten et al. [1] that can be unbiased using the IPS formula and discuss what is the property of that loss function that allows for IPS correction. (15.0p)

The performance metric that is optimized in Thorsten et al. is the sum of the ranks of the relevant results. It can be written as follows:
$$ \Delta(\mathbf{y}|\mathbf{x}_i, r_i) = \sum_{y \in \mathbf{y}}\text{rank}(y|\mathbf{y}) \cdot r_i(y) $$
where $\mathbf{x}_i$ is the vector representation of query $i$, $\mathbf{y}$ is a ranking of documents $y$, and $r_i(y) $ is the relevance of a specific document with respect to the query. These are binary values $r_i(y) \in \{0, 1\}$. This function can be unbiased because it is linearly decomposable and dependent only on relevant items. We can see the bias in the loss when calculating its expectation over the distribution of probabilities of a document being observed:

$$ \mathbb{E}_o[\Delta(\mathbf{y}|\mathbf{x}_i, r_i)] = \mathbb{E}_o\left [\sum_{y \in \mathbf{y}}\text{rank}(y|\mathbf{y}) \cdot r_i(y)\right ] = \\
\mathbb{E}_o\left [\sum_{y: o_y = 1 \land r_i(y) = 1}\text{rank}(y|\mathbf{y})\right] = \sum_{y: r_i(y) = 1}P(o_y = 1 | \mathbf{y}, y) \cdot \text{rank}(y|\mathbf{y})$$

where the $P(o_y = 1)$ is the probability of observation of specific document $y$. To debias, the metric that is optimized in Joachims et al. therefore is:

$$ \Delta_{IPS}(\mathbf{y}|\mathbf{x}_i, \mathbf{\bar{y}_i}, o_i) = \sum_{y:o_i(y)=1}\frac{\text{rank}(y|\mathbf{y}) \cdot r_i(y)}{Q(o_i(y)=1|\mathbf{x}_i, \mathbf{\bar{y}_i}, o_i)} $$

where $\mathbf{\bar{y}_i}$ is the ranking presented to the user, $o_i(y)$ is a binary variable indicating whether the document $y$ is observed.

> b. Try to provide an IPS corrected formula for each of the three LTR loss functions that you have seen and implemented in the computer assignment. If a loss function cannot be adapted in the IPS formula, discuss the possible reasons.

To debias a metric using the IPS method, it must be linearly decomposible, i.e. it must be able to be rewritten as 
$$ \Delta(f_{\theta}, D, y) = \sum_{d_i \in D} \lambda(\text{rank}(d_i | f_{\theta}, D)) \cdot y(d_i), $$
where $f_{\theta}$ is the ranker, $D$ is the document collection, $y(d_i)$ is the ground truth relevance of document $d_i$, and $\lambda$ is a rank weighing function. To pre-emptively explain some later notation, $s_i$ is the score of document $i$ given the ranking function, and $c_i$ is a binary variable indicating whether document $i$ is clicked or not.

None of these loss functions have an IPS-corrected version. Let us discuss the reasons one by one:

1. MSE is not linearly decomposible and cannot be written in the above form (i.e, the relevance labels/click status cannot be factored out and MSE rewritten as a product of a rank weighting function and the relevance labels). The general assumption in IPS corrections is that you can limit yourself to looking at clicked (and thus relevant) documents. However, MSE needs requires a contribution from non-relevant documents, because without it the ranking function would simply learn to label all items as relevant. In this case then, there is no way to limit the error metric to all observed items because you do not know how many non-clicked items are observed. To represent this formally, you cannot you cannot write $$MSE_{IPS} = \frac{1}{N}\sum_{i:o_i = 1}\frac{||s_i - c_i||^2 \cdot c_i}{P(o_i = 1|s_i, D)}$$ because 1) you cannot refactor MSE in the form of the numerator, and 2) even if you did, your ranker would learn to label everything as relevant, as the items misclassified but not clicked do not contribute to the loss. You also cannot use $$MSE_{IPS} = \frac{1}{N}\sum_{i:o_i = 1}\frac{||s_i - c_i||^2}{P(o_i = 1|s_i, D)}$$ because you have no way of knowing for which unclicked documents $o_i = 1$ and therefore cannot evaluate the error function for all observed documents..

2. You can write pairwise loss as the following: $$ C_T = \sum_{i,j \in \mathcal{P}} C(s_i; s_j) = \sum_{i \in D}\sum_{j \in D}C(s_i, s_j) = \\
\sum_{i \in D}\sum_{j \in D} \frac{1}{2}(1- S_{ij})\sigma(s_i - s_j) + \log(1+ e^{-\sigma(s_i - s_j)}) = \sum_{i \in D}C(i),$$
where $S_{ij} = \{0, 1, -1\}$ depending on whether the relevance (and therefore click status) of document $i$ is 1) the same as document $j$ ($S_{ij} = 0$), 2) the greater than document $j$ ($S_{ij} = 1$), or 3) lower than document $j$ ($S_{ij} = -1$). $C(i)$ cannot be refactored to the desired multiplication of rank weighing and relevance status and therefore is not linearly decomposible. More intuitively, changes in the click status of one document $i$ will affect not only the contribution to the loss due to $i$ but also the contribution to the loss due to all other documents. IPS only corrects the contribution of the single document, assuming that the error contributions due to specific documents are independent of each other. However, this is not the case with pairwise loss. Debiasing would have to entail reweighing the change to all other contributions to the loss by the propensity of document $i$, not only $i$'s contribution.

3. Listwise loss is just a version of pairwise loss with rescaled gradients. This means that the reasoning above still applies, making it not possible to debias the loss without extending IPS.  


### 4 Extensions to IPS

> a. The IPS in Thorsten et al. [1] works with binary clicks. How can it be extended to the graded user feedback, e.g., a 0 to 5 rating scenario in a recommendation.

The IPS metric can be extended to a graded user feedback as follows. Let $r_i(y)$ the graded feedback such that $r_i(y) \in \{0,...,m\}$. Let our IPS be the exact same as in Joachims et al.:
$$ \Delta_{IPS}(\mathbf{y}|\mathbf{x}_i, r_i) =\sum_{y \in \mathbf{y}}\frac{\text{rank}(y|\mathbf{y}) \cdot r_i(y)}{P(o_y = 1 | \mathbf{y}, y)},$$

Then in expectation:

$$ \mathbb{E}_o[\Delta_{IPS}(\mathbf{y}|\mathbf{x}_i, r_i)] = \mathbb{E}_o\left [\sum_{y \in \mathbf{y}}\frac{\text{rank}(y|\mathbf{y}) \cdot r_i(y)}{P(o_y = 1 | \mathbf{y}, y)}\right ] = \\
\mathbb{E}_o\left [\sum_{y: o_y = 1 \land r_i(y) > 0} \frac{\text{rank}(y|\mathbf{y}) \cdot r_i(y)}{P(o_y = 1 | \mathbf{y}, y)} \right] = \sum_{y: r_i(y) > 0} \frac{P(o_y = 1 | \mathbf{y}, y) \cdot r_i(y) \cdot \text{rank}(y|\mathbf{y})}{P(o_y = 1 | \mathbf{y}, y)} = \\
\sum_{y \in \mathbf{y}} r_i(y) \cdot \text{rank}(y|\mathbf{y}) = \Delta(\mathbf{y}|\mathbf{x}_i, r_i).$$

It can be noted that it is easier to calculate propensities in this graded context, as feedback is typically much more explicit/less noisy than click feedback. For example, in movie recommendations, the propensities may simply be a result of star ratings of users.

> b. One of the issues with IPS is its high variance. Explain the issue and discuss what can be done to reduce the variance of IPS.

High variance is a common problem in IPS and is often due to 1) not enough training data, 2) extremely strong position bias and therefore small propensities, or 3) large click noise on many documents. Low propensities will cause large variations in the unbiased loss metric, resulting in sub-optimal optimization. A way to resolve this is to introduce propensity clipping, in which small propensities are clipped to some lower bound, thus mitigating the impact of strong propensities. This causes tradeoff between variance and bias, adding some bias to significantly reduce variance. In this context, given a lower bound $\tau$ we can write:

$$ \Delta_{IPS}(\mathbf{y}|\mathbf{x}_i, r_i) =\sum_{y \in \mathbf{y}}\frac{\text{rank}(y|\mathbf{y}) \cdot r_i(y)}{\max\{\tau, P(o_y = 1 | \mathbf{y}, y)\}}$$

