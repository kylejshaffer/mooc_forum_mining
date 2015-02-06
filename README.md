<h1>Independent Study in MOOC Forum Mining</h1>
<h2>Study Background</h2>
<p>This repo contains code and technical documentation for a research project in classifying posts within MOOC discussion forums. The MOOC was offered as a course through UNC's School of Information and Library Science in the Fall of 2013, and was hosted on the Coursera platform.</p>
<p>The motivation for the study was to provide an experimental basis for an automated tool to alert instructors to posts within forums that may warrant manual intervention. MOOCs are extremely popular in their initial enrollment with thousands of students enrolling in courses in a very short amount of time. This makes course management extremely challenging, and this challenge presents an opportunity for automated machine learning tools to help predict which posts instructors should focus on. More details are presented in the paper, located at <code>paper_latex_files/shaffer_mooc_study.pdf</code></p>

<h2>Files and Repo Structure</h2>
<p>There are several directories within the repo containing different types of files necessary for the study. Each is detailed below. To get a better sense of the background for the study and the results, you can take a look at the paper in <code>paper_latex_files</code> and read the paper there. <code>Latex</code> files are also included as well. Additionally, HTML, JavaScript, and CSS files for building the data collection interface used in the study can be found in the <code>interface</code> directory. Finally, the <code>code</code> directory contains Python code that was used for manipulating the raw forum data, extracting and engineering features, and running machine learning experiments.</p>

<h3>Paper Files</h3>
<b>Relevant files:</b>
<ul>
<li><code>shaffer_mooc_study.pdf</code></li>
<li><code>shaffer_mooc_study.tex</code></li>
</ul>
<h3>Interface</h3>
<p>This directory contains HTML, JavaScript and CSS used for building the data collection interface used in this study. Relevant files are detailed below.</p>
<ul>
<li><code>index.html</code>: Main HTML interface MTurk workers used to annotate our dataset.</li>
<li><code>instructions.html</code>: HTML file with instructions given to MTurk workers on how to annotate the dataset and the definitions that would be used for the class labels we needed to collect.</li>
<li><code>thread.html</code>: HTML file presenting MTurk workers with individual thread and outlined post to be annotated.</li>
</ul>

<h3>Code</h3>
<p>This directory contains two sub-directories: one for processing data and constructing features, and one for running machine learning experiments and the ablation analysis in the paper</p>
<b>Data Processing</b>
<ul>
<li><code>ablation_data_prep.py</code>: script for combining previoiusly computed features, cleaning up the constructed dataset, and constructing train-test pairs for running machine learning ablation analysis.</li>
<li>code>feature_extractor.py</code>: code for processing raw data and extracting relevant features from data. Many of these were used in the <b>raw features</b> section in the paper in addition to <b>LIWC</b> linguistic count features.</li>
<li><code>liwc_text.py</code>: code for extracting only text, removing markup and punctuation for use with LIWC software.</li>
<li><code>mooc_datareader.py</code>: early script for reading in Excel version of dataset and converting it to JSON.</li>
</ul>
<b>Machine Learning</b>
<ul>
<li><code>ablation_analysis.py</code>: code used to run machine learning experiments using Logistic Regression and evaluating models with <b>Average Precision</b> over 10-fold cross validation, as well as a feature ablation analysis. This code generated the results section of the paper.</li>
<li><code>example_classifier.py</code>: example script used early on to explore classifiers. Modified from an <a href="http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html">original post</a> in the docs for Scikit-Learn.</li>
<li></li>
<li></li>
</ul>
