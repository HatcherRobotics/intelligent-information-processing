#### 1、 分析为什么平方损失函数不适用于分类问题。

> 最小化平方损失函数本质上等同于在误差服从高斯分布的假设下的极大似然估计，在分类问题下大部分时候误差并不服从高斯分布。更直观地说，平方损失函数是通过真实值与预测值间的距离反映优化的程度，而在分类问题中常用one-hot的形式进行编码，其预测值与真实值间的距离没有实际意义。
>
> 
>
> #### 2、 假设有$N$个样本$x(1),x (2), ···,x(N)$服从正态分布$N(\mu,\sigma^2)$,其中$\mu$未知。
>
> (1)使用最大似然估计来求解最优参数$\mu^{ML}$。
>
> $$\because x(1),x (2), ···,x(N) \sim N(\mu,\sigma^2)$$
> $$\therefore p(x_i|\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\cdot exp \{-\frac{(x_i-\mu)^2}{2\sigma^2}\}$$
> $$\therefore p(\pmb{x}|\mu,\sigma)=\prod_{i=1}^{N}p(x_i|\mu,\sigma)={(\frac{1}{\sqrt{2\pi}\sigma}})^N \cdot \prod_{i=1}^N exp\{-\frac{(x_i-\mu)^2}{2\sigma^2}\}$$
>
> $$ \therefore log[p(\pmb{x}|\mu,\sigma)]=Nlog\frac{1}{\sqrt{2\pi}\sigma}+ \sum_{i=1}^{N}- \frac{(x_i-\mu)^2}{2\sigma^2}$$
>
> $$\because \mu^{ML}= \min\limits_{\mu}log[p(\pmb{x}|\mu,\sigma)]$$
>
> $$s.t. \frac{\partial{log[p(\pmb{x}|\mu,\sigma)]}}{\partial{\mu}}=0$$
>
> $$\therefore \sum_{i=1}^{N}[- \frac{2(x_i-\mu)}{2\sigma^2} \cdot(-1)]=0$$
>
> $$\therefore \sum_{i=1}^{N}(x_i-\mu)=0$$
>
> $$\therefore \sum_{i=1}^{N}x_i-N\mu=0$$
>
> $$\therefore \mu = \frac{1}{N}\sum_{i=1}^{N}x_i$$
>
> $$\therefore \mu^{ML}= \frac{1}{N}\sum_{i=1}^{N}x_i$$

##### 
##### (2)若参数$\mu$为随机变量，并服从正态分布 $N(\mu_0,\sigma_0^2)$，使用最大后验估计来求解最优参数$\mu^{MAP}$。

#### 3、 在习题2-6中，证明当$N→∞$ 时，最大后验估计趋向于最大似然估计。

#### 4、 在 Softmax 回归的风险函数（公式 (3.39)）中，如果去掉正则化项会有什么影响？

> 会导致参数矩阵$W$中，对应每个类别的矩阵向量都非常大。
####5、利用数学仿真验证习题4的结论