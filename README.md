# vec2rec

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

[![Build Status](https://travis-ci.com/csci-e-29/2020sp-vec2rec-crispin-nosidam.svg?token=<travis-token>&branch=master)](https://travis-ci.com/csci-e-29/2020sp-vec2rec-crispin-nosidam)

[![Maintainability](https://api.codeclimate.com/v1/badges/<cc-badge>/maintainability)](https://codeclimate.com/repos/<cc-repo>/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/<cc-badge>/test_coverage)](https://codeclimate.com/repos/<cc-repo>/test_coverage)

##Goal
All of us looked for a job before. We also search training courses to equip ourselves for the job market. Some of us also try to find suitable candidates for their team or try to train up their staff. What challenges do these tasks have in common? With limited time to search, we have:
1.	An explosive amount of choices
2.	Choices that are mostly similar to each other
3.	Data, while somewhat categorized, the differentiators are in words, i.e.: natural languages
4.	Different preferences with different background – the best paid Java programmer position may be easy to search, but are you up to it?
A description of yourself, i.e.: a resume, is ultimately what gets you into an interview, after the broad-stroke categorizations like “Java programmer”. Keyword search engines like google are common, but what if you can search with your resume?
Similarity, a search with job description for the best candidate, or the relevancy of a training to a job we want would be nice.
Vec2rec is a recommendation engine that, with natural language processing, enable us to search with a description, be it a resume, a job description or a training description for most relevant results.
In addition, we can also run **what-if scenarios**, e.g.: with my current resume, how much closer would it be for me to a dream job if I take this training course? 

##Use Cases
* Job Seekers
  * Find the most suitable jobs
* Headhunters
  * Find the most suitable candidates for a job
* Training Professionals / Managers / Job Seekers
  * Most relevant trainings to a job
  * What jobs a training can enable
  * Increment of a candidate’s suitability for a job after a training

##Architecture
![alt text](https://github.com/crispin-nosidam/2020sp-vec2rec-crispin-nosidam/tree/master/vec2rec/images/arch_diag.png "Vec2Rec Architecture Diagram")

