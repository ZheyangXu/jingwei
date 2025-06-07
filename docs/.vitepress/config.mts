import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "精卫",
  description:
    "又北二百里，曰发鸠之山，其上多柘木，有鸟焉，其状如乌，文首，白喙，赤足，名曰：“精卫”，其鸣自詨。是炎帝之少女，名曰女娃。女娃游于东海，溺而不返，故为精卫，常衔西山之木石，以堙于东海。漳水出焉，东流注于河。 《山海经·北山经》",
  base: "/jingwei/",
  themeConfig: {
    logo: "logo.png",
    // https://vitepress.dev/reference/default-theme-config
    // top-left nav bar
    nav: [
      { text: "主页", link: "/" },
      {
        text: "强化学习",
        link: "/reinforcement_learning",
        items: [
          { text: "强化学习基础", link: "/reinforcement_learning/basic" },
          { text: "深度强化学习", link: "/reinforcement_learning/deeprl" },
          { text: "离线强化学习", link: "/reinforcement_learning/offlinerl" },
        ],
      },
      { text: "文档", link: "/docs" },
      { text: "API", link: "/api" },
    ],

    sidebar: {
      "/reinforcement_learning/": [
        {
          text: "强化学习",
          items: [
            {
              text: "强化学习基础",
              link: "/reinforcement_learning/basic",
              items: [
                {
                  text: "马尔可夫决策过程",
                  link: "/reinforcement_learning/basic/mdp",
                },
                {
                  text: "Value-based Methods",
                  link: "/reinforcement_learning/basic/value_based",
                  items: [
                    {
                      text: "值迭代",
                      link: "/reinforcement_learning/basic/value_based/value_iteration",
                    },
                    {
                      text: "多臂老虎机",
                      link: "/reinforcement_learning/basic/value_based/multi_armed_bandit",
                    },
                    {
                      text: "时序差分",
                      link: "/reinforcement_learning/basic/value_based/td",
                    },
                    {
                      text: "多步时序差分",
                      link: "/reinforcement_learning/basic/value_based/n_step_td",
                    },
                    {
                      text: "MCTS",
                      link: "/reinforcement_learning/basic/value_based/mcts",
                    },
                    {
                      text: "Q-learning",
                      link: "/reinforcement_learning/basic/value_based/q_learning",
                    },
                  ],
                },
                {
                  text: "基于策略的方法",
                  link: "/reinforcement_learning/basic/policy_based",
                  items: [
                    {
                      text: "策略迭代",
                      link: "/reinforcement_learning/basic/policy_based/policy_iteration",
                    },
                    {
                      text: "策略梯度",
                      link: "/reinforcement_learning/basic/policy_based/policy_gradients",
                    },
                    {
                      text: "Actor Critic",
                      link: "/reinforcement_learning/basic/policy_based/actor_critic",
                    },
                  ],
                },
                {
                  text: "MDPs 的建模与抽象",
                  link: "/reinforcement_learning/basic/modeling_and_abstraction_for_MDPs",
                },
              ],
            },
            {
              text: "深度强化学习",
              link: "/reinforcement_learning/deeprl",
            },
            {
              text: "多智能体强化学习",
              link: "/reinforcement_learning/multi_agent_rl",
              items: [
                {
                  text: "标准形式博弈",
                  link: "/reinforcement_learning/multi_agent_rl/normal_form_game",
                },
                {
                  text: "扩展形式博弈",
                  link: "/reinforcement_learning/multi_agent_rl/extensive_form_game",
                },
                {
                  text: "多智能体强化学习",
                  link: "/reinforcement_learning/multi_agent_rl/multi_agent_rl",
                },
                {
                  text: "多能体博弈的建模与抽象",
                  link: "/reinforcement_learning/multi_agent_rl/multi_agent_games",
                },
              ],
            },
          ],
        },
      ],
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/ZheyangXu/jingwei" },
    ],
    footer: {
      message:
        'Released under the <a href="https://github.com/ZheyangXu/jingwei-docs/main/LICENSE">MIT License</a>.',
      copyright:
        'Copyright © 2024-present <a href="https://github.com/ZheyangXu/jingwei">ZheyangXu</a>',
    },
  },
  markdown: {
    math: true,
    lineNumbers: true,
  },
  mermaid: {},
});
