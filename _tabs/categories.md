---
# the default layout is 'page'
icon: fas fa-stream
order: 1
description: Blog categories
---

<div class="category-hub" markdown="1">

# Blog Categories

<p class="category-hub__intro">A home for my course notes, algorithm practice, and longer technical writeups. The cards below are ready for future posts, so each project can grow into its own collection over time.</p>

<div class="category-grid">
<a class="category-card" href="#stanford-cs336">
<div class="category-card__media category-card__media--image" style="background-image: url('/assets/img/cs336.png');"></div>
<div class="category-card__body">
<h2>Stanford CS336: LLM from Scratch</h2>
<p>Stanford CS336 systematically explains how to train large language models (LLMs) from scratch, covering Tokenizers, Transformer architectures and variants, model training, distributed systems (4D Parallel), inference optimization, Scaling Laws, model evaluation, data generation, RLHF, RLVR, and other core topics. This blog series follows the course materials while analyzing engineering implementation details and PyTorch code, supplemented with hands-on reproduction and personal reflections to help readers build a comprehensive understanding of modern LLM training and inference workflows.</p>
<div class="category-card__meta">
<span class="category-pill">level: advanced</span>
<span class="category-pill">0 posts</span>
</div>
</div>
</a>

<a class="category-card" href="#leetcode">
<div class="category-card__media category-card__media--image" style="background-image: url('/assets/img/algorithm.png');"></div>
<div class="category-card__body">
<h2>Algorithm: Leetcode problems</h2>
<p>This section organizes Leetcode practice around mainstream algorithms and data structures, including binary search, greedy strategies, arrays, linked lists, stacks, queues, hash tables, trees, graphs, dynamic programming, and sliding window techniques. Each note focuses on the core idea behind a pattern, the conditions under which it applies, common edge cases, and clean implementation details, with the goal of turning scattered problem solving into a structured algorithm toolkit.</p>
<div class="category-card__meta">
<span class="category-pill">practice</span>
<span class="category-pill">0 posts</span>
</div>
</div>
</a>
</div>

<section id="stanford-cs336" class="post-slot" markdown="1">

## Stanford CS336: LLM from Scratch

{% assign cs336_posts = site.categories['Stanford CS336'] %}
{% if cs336_posts.size > 0 %}
<ul class="post-list-mini">
{% for post in cs336_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
No posts yet. Future posts in this area can use `categories: [Stanford CS336]`.
{% endif %}

</section>

<section id="leetcode" class="post-slot" markdown="1">

## Algorithm: Leetcode problems

{% assign leetcode_posts = site.categories['Leetcode'] %}
{% if leetcode_posts.size > 0 %}
<ul class="post-list-mini">
{% for post in leetcode_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
No posts yet. Future posts in this area can use `categories: [Leetcode]`.
{% endif %}

</section>

</div>
