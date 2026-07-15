(function () {
  const STORAGE_KEY = 'site-language';
  const LANGUAGES = {
    zh: {
      htmlLang: 'zh-CN',
      toggleLabel: 'EN',
      toggleTitle: 'Switch to English',
      title: 'Zhiyuan Guo',
      subtitle: '学习笔记 / 项目实践 / 生活记录',
      searchPlaceholder: '搜索...',
      searchCancel: '取消',
      footerRights: '保留部分权利。',
      footerTooltip: '除非另有说明，本网站上的博客文章均由作者按照知识共享署名 4.0 国际 (CC BY 4.0) 许可协议进行授权。',
      themeCreditPrefix: '本站采用',
      themeCreditTheme: '主题',
      updateNotice: '发现新内容',
      updateButton: '更新',
      tabs: {
        '/': '首页',
        '/categories/': '分类',
        '/tags/': '标签',
        '/archives/': '归档',
        '/about/': '关于'
      }
    },
    en: {
      htmlLang: 'en',
      toggleLabel: '中',
      toggleTitle: '切换到中文',
      title: 'Zhiyuan Guo',
      subtitle: 'Personal Blog & Notes',
      searchPlaceholder: 'Search...',
      searchCancel: 'Cancel',
      footerRights: 'Some rights reserved.',
      footerTooltip: 'Except where otherwise noted, the blog posts on this site are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).',
      themeCreditPrefix: 'Using',
      themeCreditTheme: 'theme',
      updateNotice: 'New content available',
      updateButton: 'Update',
      tabs: {
        '/': 'Home',
        '/categories/': 'Categories',
        '/tags/': 'Tags',
        '/archives/': 'Archives',
        '/about/': 'About'
      }
    }
  };

  function normalizePath(pathname) {
    const path = pathname.replace(/\/index\.html$/, '/');
    return path.endsWith('/') ? path : `${path}/`;
  }

  function preferredLanguage() {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (saved === 'zh' || saved === 'en') {
      return saved;
    }

    return document.documentElement.lang.toLowerCase().startsWith('zh') ? 'zh' : 'en';
  }

  function setText(selector, value) {
    const element = document.querySelector(selector);

    if (element) {
      element.textContent = value;
    }
  }

  function updateTabs(copy) {
    document.querySelectorAll('#sidebar .nav-link').forEach((link) => {
      const path = normalizePath(new URL(link.getAttribute('href'), window.location.origin).pathname);
      const label = link.querySelector('span');

      if (label && copy.tabs[path]) {
        label.textContent = copy.tabs[path];
      }
    });
  }

  function updateCurrentPageLabels(copy) {
    const path = normalizePath(window.location.pathname);
    const label = copy.tabs[path] || copy.title;

    setText('#breadcrumb span', label);
    setText('#topbar-title', label);

    document.title = path === '/' ? copy.title : `${label} | ${copy.title}`;
  }

  function updateFooter(copy) {
    const rights = document.querySelector('footer [data-bs-toggle="tooltip"]');

    if (rights) {
      rights.textContent = copy.footerRights;
      rights.setAttribute('title', copy.footerTooltip);
      rights.setAttribute('data-bs-original-title', copy.footerTooltip);
    }

    const credit = document.querySelector('footer p:last-child');

    if (credit) {
      const links = credit.querySelectorAll('a');

      if (links.length >= 2) {
        credit.replaceChildren(
          document.createTextNode(`${copy.themeCreditPrefix} `),
          links[0],
          document.createTextNode(` ${copy.footerTheme || copy.themeCreditTheme} `),
          links[1]
        );
      }
    }
  }

  function updateSearch(copy) {
    const searchInput = document.querySelector('#search-input');

    if (searchInput) {
      searchInput.setAttribute('placeholder', copy.searchPlaceholder);
    }

    setText('#search-cancel', copy.searchCancel);
  }

  function updateNotification(copy) {
    setText('#notification .toast-body', copy.updateNotice);
    setText('#notification .btn-primary', copy.updateButton);
  }

  function updateToggle(button, lang, copy) {
    const label = button.querySelector('.language-toggle__label');

    if (label) {
      label.textContent = copy.toggleLabel;
    }

    button.setAttribute('aria-label', copy.toggleTitle);
    button.setAttribute('title', copy.toggleTitle);
    button.dataset.language = lang;
  }

  function applyLanguage(lang, button) {
    const copy = LANGUAGES[lang];

    document.documentElement.lang = copy.htmlLang;
    setText('.site-title', copy.title);
    setText('.site-subtitle', copy.subtitle);
    updateTabs(copy);
    updateCurrentPageLabels(copy);
    updateSearch(copy);
    updateFooter(copy);
    updateNotification(copy);
    updateToggle(button, lang, copy);
  }

  function createToggle() {
    const sidebarBottom = document.querySelector('#sidebar .sidebar-bottom');

    if (!sidebarBottom || document.querySelector('.language-toggle')) {
      return null;
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-link nav-link language-toggle';
    button.innerHTML = '<i class="fa-solid fa-language" aria-hidden="true"></i><span class="language-toggle__label"></span>';

    const divider = sidebarBottom.querySelector('.icon-border');
    sidebarBottom.insertBefore(button, divider || sidebarBottom.firstChild);

    button.addEventListener('click', () => {
      const nextLang = button.dataset.language === 'zh' ? 'en' : 'zh';
      localStorage.setItem(STORAGE_KEY, nextLang);
      applyLanguage(nextLang, button);
    });

    return button;
  }

  document.addEventListener('DOMContentLoaded', () => {
    const button = createToggle();

    if (button) {
      applyLanguage(preferredLanguage(), button);
    }
  });
})();
