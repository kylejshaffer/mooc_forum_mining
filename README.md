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
