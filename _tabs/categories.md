---
# the default layout is 'page'
icon: fas fa-stream
order: 1
description: Blog categories
---

<div class="category-hub" markdown="1">

# Blog Categories

<p class="category-hub__intro">A home for my notes, reading, projects, and longer technical writeups. The cards below are ready for future posts, so each area can grow into its own collection over time.</p>

<div class="category-grid">
<a class="category-card" href="#llm-notes">
<div class="category-card__media">LLM</div>
<div class="category-card__body">
<h2>LLM Notes</h2>
<p>Notes on language models, transformers, training systems, evaluation, and papers I am reading.</p>
<div class="category-card__meta">
<span class="category-pill">level: ongoing</span>
<span class="category-pill">0 posts</span>
</div>
</div>
</a>

<a class="category-card" href="#projects">
<div class="category-card__media">DEV</div>
<div class="category-card__body">
<h2>Projects</h2>
<p>Build logs, implementation notes, debugging stories, and small systems I want to document properly.</p>
<div class="category-card__meta">
<span class="category-pill">practice</span>
<span class="category-pill">0 posts</span>
</div>
</div>
</a>

<a class="category-card" href="#study-notes">
<div class="category-card__media">LOG</div>
<div class="category-card__body">
<h2>Study Notes</h2>
<p>Course notes, summaries, and compact explanations for concepts I want to keep close at hand.</p>
<div class="category-card__meta">
<span class="category-pill">notes</span>
<span class="category-pill">0 posts</span>
</div>
</div>
</a>
</div>

<section id="llm-notes" class="post-slot" markdown="1">

## LLM Notes

{% assign llm_posts = site.categories['LLM Notes'] %}
{% if llm_posts.size > 0 %}
<ul class="post-list-mini">
{% for post in llm_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
No posts yet. Future posts in this area can use `categories: [LLM Notes]`.
{% endif %}

</section>

<section id="projects" class="post-slot" markdown="1">

## Projects

{% assign project_posts = site.categories['Projects'] %}
{% if project_posts.size > 0 %}
<ul class="post-list-mini">
{% for post in project_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
No posts yet. Future posts in this area can use `categories: [Projects]`.
{% endif %}

</section>

<section id="study-notes" class="post-slot" markdown="1">

## Study Notes

{% assign study_posts = site.categories['Study Notes'] %}
{% if study_posts.size > 0 %}
<ul class="post-list-mini">
{% for post in study_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
No posts yet. Future posts in this area can use `categories: [Study Notes]`.
{% endif %}

</section>

</div>
