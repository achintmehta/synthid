# Plain vs watermarked: Qwen/Qwen2.5-1.5B-Instruct, 16 keys, seed 0

Score = mean g-value over the scored tokens. 0.5 means no watermark.

| Prompt | Plain | Watermarked | Watermarked, wrong keys | Shared opening | Mean entropy (plain / wm) |
|---|---|---|---|---|---|
| story | 0.508 | 0.621 | 0.503 | 0 tokens | 2.12 / 1.87 |
| fact | 0.514 | 0.535 | 0.487 | 15 tokens | 0.84 / 0.47 |
| code | 0.492 | 0.565 | 0.516 | 2 tokens | 0.18 / 0.66 |

### story: Write a short story about a lighthouse keeper who finds a message in a bottle.

**Plain** (score 0.508)

> John had always been fascinated with the stories of lighthouses and their important role in guiding seafarers safely into ports. As an adult, he worked as a lifeguard at an on-shore beach while dreaming of one day becoming the lighthouse keeper for his favorite island. One sunny morning, John was assigned to patrol the coast ahead of the first wave rolling onto shore.
> 
> Suddenly, the wind began to howl and rain fell from above. The sea roared like a beast until the lookout shouted over the storm saying that waves were rising rapidly and they could not be seen far below them. 
> 
> It's when another voice yelled "Look out - there’s debris ahead!" As the scene before him unfolded, something caught his interest and he found himself asking questions where none existed. The question became: What did you write? Why so big?
> 
> His eyes focused back onto the small piece of paper lying inside a plastic bag attached to the end of twine floating on the ocean surface. It

**Watermarked** (score 0.621, wrong keys 0.503)

> Once upon a time, there was a man named Tom who worked at the lighthouse on the small island off the coast of Britain. It had been over two decades since his parents had built it and they died on the job, but he loved it, so much. The keepers would help maintain the beacon for others when they came to the islands.
> One evening, as he was getting ready to go home each night after staying up all night long working at the light house, he got wind of something exciting. There was someone at the gate of the lighthouse with a bottle of water that he could only imagine what happened inside.
> So, Tom made his way through the door and asked him: "Can ya offer me a drink?" The man turned around and he offered Tom a bottle of water that contained an unadulterated letter addressed to himself.
> Without hesitation, Tom took the bottle and read it carefully. He said words like 'thank you,' 'hello,' 'good morning/'

### fact: What year did the Berlin Wall fall, and why did it matter? Answer in two sentences.

**Plain** (score 0.514)

> The Berlin Wall fell on November 9, 1989. Its fall was significant because it symbolized the end of one of the most divisive periods of the Cold War, fostering hope for global peace and unity that lasted even as Germany reunified into a single country.

**Watermarked** (score 0.535, wrong keys 0.487)

> The Berlin Wall fell on November 9, 1989. This event was significant as it symbolized the end of Communist rule and marked a turning point that led to the reunification of Germany.

### code: Write a Python function that returns the n-th Fibonacci number iteratively.

**Plain** (score 0.492)

> Certainly! The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. Here's a Python function to calculate the nth Fibonacci number iteratively:
> 
> ```python
> def fibonacci(n):
>     """
>     Returns the n-th Fibonacci number using an iterative approach.
>     
>     :param n: Index (n >= 0) of the desired Fibonacci number in the sequence
>     :return: The n-th Fibonacci number
>     
>     >>> fibonacci(0)
>     0
>     >>> fibonacci(1)
>     1
>     >>> fibonacci(2)
>     1
>     >>> fibonacci(3)
>     2
>     >>> fibonacci(5)
>     5
>     >>> fibonacci(8)
>     21
>     """
>     if n == 0:
>         return 0
>     elif n == 1:
>         return 1
> 
>     a, b = 0, 1
>     for _ in

**Watermarked** (score 0.565, wrong keys 0.516)

> Certainly! Iterative approach is often preferred for problems involving sequence generation because it doesn't use recursion or call stacks which can lead to stack overflow errors (particularly when calculating large numbers of Fibonacci sequences with depth).
> 
> Below is a Python function that calculates the `n`-th Fibonacci number using an iterative method:
> 
> ```python
> def fibonacci_iterative(n):
>     if n <= 0:
>         return "Input should be a positive integer."
>     elif n == 1:
>         return 0
>     else:
>         prev = 0
>         curr = 1
> 
>         # Calculate Fibonacci numbers up to n=50 since this seems high enough without issues.
>         # The last two values will define what follows from there.
>         for _ in range(2, n+1):
>             next_val = prev + curr
>             prev, curr = curr, next_val
> 
>         return curr
> 
> # Example usage:
> print(fibonacci_iterative(10))  # Output: 3
