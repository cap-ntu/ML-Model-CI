<p align="center"> <img src="docs/img/iconv1.svg" width="230" alt="..."> </p>

<h1 align="center">
    Machine Learning Model CI - 中文简介
</h1>

<p align="center">
    <a href="https://www.python.org/downloads/release/python-370/" title="python version"><img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg"></a>
    <a href="https://travis-ci.com/cap-ntu/ML-Model-CI" title="Build Status"><img src="https://travis-ci.com/cap-ntu/ML-Model-CI.svg?token=SvqJmaGbqAbwcc7DNkD2&branch=master"></a>
    <a href="https://app.fossa.com/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI?ref=badge_shield" title="FOSSA Status"><img src="https://app.fossa.com/api/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI.svg?type=shield"></a>
    <a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cap-ntu/ML-Model-CI&amp;utm_campaign=Badge_Grade" title="Codacy Badge"><img src="https://app.codacy.com/project/badge/Grade/bfb9f8b11d634602acd8b67484a43318"></a>
    <a href="https://codebeat.co/a/yizheng-huang/projects/github-com-cap-ntu-ml-model-ci-master"><img alt="codebeat badge" src="https://codebeat.co/badges/343cc340-21c6-4d34-ae2c-48a48e2862ba" /></a>
    <a href="https://github.com/cap-ntu/ML-Model-CI/graphs/commit-activity" title="Maintenance"><img src="https://img.shields.io/badge/Maintained%3F-YES-yellow.svg"></a>
    <a href="https://gitter.im/ML-Model-CI/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge" title="Gitter"><img src="https://badges.gitter.im/ML-Model-CI/community.svg"></a>
</p>

<p align="center">
    <a href="README.md">English Version</a> •
    <a href="#系统简介">系统简介</a> •
    <a href="#简易安装">简易安装</a> •
    <a href="#快速使用">快速使用</a> •
    <a href="#详细教程">详细教程</a> •
    <a href="#加入我们">加入我们</a> •
    <a href="#文献引用">文献引用</a> •
    <a href="#版权许可">版权许可</a>
</p>

## 系统简介

Machine Learning Model CI 是一个**云上一站式机器学习模型和服务运维平台**，旨在解决模型训练完成后，到上线成为服务的”最后一公里问题“ -- 在训练得到模型和线上机器学习应用之间，构建了一个高度自动化的桥梁。

系统目前正处于快速迭代开发中，目前我们提供了如下功能，用户 1）可以注册模型到我们的系统，享受自动化的一揽子服务；2）也可以分别使用各个功能

1. **模型管家.** 该模块接受用户注册的原始训练模型，将其存储到一个中心化的数据库当中。并提供了若干API帮助用户在本地修改，检索，删除模型。
2. **模型转换.** 在收到用户的注册请求后，模型会被自动优化和转化为高性能的部署格式。目前支持的格式有Tensorflow SavedModel, ONNX, TorchScript, TensorRT。
3. **模型解析评估.** 为了保证高质量的线上服务，上线之前的模型需要大量的性能评估测试，一方面给模型性能调优提供参考，另一方面给线上服务设置提供参考。我们的评估模块可以对模型，硬件设施，软件设施进行基准评测，提供了p99延迟，吞吐等多维度的指标。
4. **模型分发上线.** 研究环境和生产环境一般是不同的，同时模型需要和模型推理服务引擎进行绑定进行服务。该模块将用户转换后的模型与各类引擎进行绑定，然后打包成docker服务，可以快速部署上线。
5. **流程控制调度.** 我们提供了一个调度器，一方面控制整个流程的自动化实施，另一方面会将各种模型转化、解析评估等任务，分发到较为空闲机器，提高集群的利用率，让整个流程更高效安全

下面若干个功能正处于测试状态，马上会在下一个版本推出，读者可以到issue中和我们进行讨论。

- [ ] **模型优化.** 我们希望将模型量化、剪枝等加入到我们的自动化管道中。
- [ ] **模型可视化微调优** 我们希望用户可以零代码的查看和调优团队中的模型。

我们非常欢迎感兴趣的同学加入到我们的开发，请联系
> *huaizhen001 AT e.ntu.edu.sg*

## 简易安装

## 快速使用

## 详细教程

## 加入我们

## 文献引用

## 版权许可