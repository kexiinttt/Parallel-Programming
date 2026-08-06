# Parallel Programming Workflow

* Identify compute intensive parts (profile)
* Implement scalable algorithms
* Optimize data arrangements to maximize locality
* Performance Tunine

---

# Parallel Computing Challenges

* Unbalanced distribution &rarr; the thread that takes the longest time to finish
* Memory bandwidth
* Data access conflicts &rarr; cause serialization and delay

---

# Parallel Speedup Rate (Amdahl’s Law)

If an application has:
* `t` sequential execution time
* `p` fraction of the execution is parallelizable
* `s` speedup achieved on the parallelizable part

Then the overall speedup rate is:

$$
t_{\text{parallel}}
= (1-p)t + \frac{pt}{s}
= \left(1-p+\frac{p}{s}\right)t
$$

$$
\text{Speedup}
= \frac{t_{\text{sequential}}}{t_{\text{parallel}}}
= \frac{t}{\left(1-p+\frac{p}{s}\right)t}
= \frac{1}{1-p+\frac{p}{s}}
$$

When \(s \to \infty\):

$$
\text{Speedup}_{\max}
= \frac{1}{1-p}
$$

So the **overall speedup rate is limited by the portion of parallelism**.