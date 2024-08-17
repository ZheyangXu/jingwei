import { defineConfig } from 'vitepress'
import { withMermaid } from "vitepress-plugin-mermaid"

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "精卫",
  description: "又北二百里，曰发鸠之山，其上多柘木，有鸟焉，其状如乌，文首，白喙，赤足，名曰：“精卫”，其鸣自詨。是炎帝之少女，名曰女娃。女娃游于东海，溺而不返，故为精卫，常衔西山之木石，以堙于东海。漳水出焉，东流注于河。 《山海经·北山经》",
  base: '/jingwei/',
  themeConfig: {
    logo: "logo.png",
    // https://vitepress.dev/reference/default-theme-config
    // top-left nav bar
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Examples', items: [
        { text: 'model free', link: '/examples/model_free'},
        { text: 'model_base', link: '/examples/model_base'}
      ] },
      { text: 'Docs', link: '/docs'},
      { text: 'API', link: '/api'},
    ],

    sidebar: {
      '/examples/': [
        { text: 'Examples', 
          items: [
            { text: 'model free', link: '/examples/model_free'},
            { text: 'model_base', link: '/examples/model_base'}
          ]
        }, 
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/ZheyangXu/jingwei' }
    ],
    footer: {
      message: 'Released under the <a href="https://github.com/ZheyangXu/jingwei-docs/main/LICENSE">MIT License</a>.',
      copyright: 'Copyright © 2024-present <a href="https://github.com/ZheyangXu/jingwei">ZheyangXu</a>'
    }
  },
  markdown: {
    math: true,
    lineNumbers: true,
  },
  mermaid: {
    
  }
})
