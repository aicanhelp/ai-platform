import js from '@eslint/js';
import globals from 'globals';
import stylistic from '@stylistic/eslint-plugin';

export default [
  {
    ignores: ['dist/**', 'build/**', 'public/**', 'charts/**', 'node_modules/**'],
  },
  js.configs.recommended,
  {
    files: ['vite.config.js'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
  {
    files: ['**/*.{js,mjs,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        ...globals.browser,
      },
    },
    plugins: {
      '@stylistic': stylistic,
    },
    rules: {
      // 禁止使用 console 输出，报错统一走 antd message api 提示
      'no-console': 'error',
      'no-unused-vars': [
        'warn',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      // 缩进 2 空格；switch 的 case 再缩进一级
      '@stylistic/indent': ['warn', 2, { SwitchCase: 1 }],
      // 语句末尾必须加分号
      '@stylistic/semi': ['warn', 'always'],
      // 禁止多余的分号（class 方法后、空语句等）；对齐 CI 的 "Extra semicolon." 检查
      '@stylistic/no-extra-semi': 'warn',
      // 多行结构的最后一项必须带拖尾逗号（单行不加）
      '@stylistic/comma-dangle': ['warn', 'always-multiline'],
      // JS 字符串统一单引号（允许为避免转义而使用双引号；JSX 属性不受此规则约束）
      '@stylistic/quotes': ['warn', 'single', { avoidEscape: true }],

      // ===== 空行与行宽 =====
      // 最多 1 个连续空行（对齐 CI 的 "There should be no multiple empty lines in function."；
      // ESLint 无“仅函数内”版本，全局生效，比 Sonar 略严）
      '@stylistic/no-multiple-empty-lines': ['warn', { max: 1 }],
      // 代码块 {} 首尾不允许空行（对齐 CI 的 "Block must not be padded by blank lines."）
      '@stylistic/padded-blocks': ['warn', 'never'],

      // ===== if/for/do/while 执行体必须加大括号 =====
      curly: ['warn', 'all'],

      // ===== 函数返回值必须一致 =====
      // "要么处处返回值，要么都不返回"：同一个函数里 return 带值和裸 return;
      // 不能混用；return 带值的函数不能隐式走到函数结尾
      'consistent-return': 'warn',

      // ===== 变量命名 =====
      // 禁止下划线命名（如 snake_case）；PascalCase 组件名与全大写常量不受影响；
      // 不检查对象属性名/解构/导入，避免后端 snake_case 字段造成误报
      camelcase: [
        'warn',
        { properties: 'never', ignoreDestructuring: true, ignoreImports: true },
      ],

      // ===== 用空格突出关键字和重要信息 =====
      // 关键字（if/for/while/function/return 等）前后留空格
      '@stylistic/keyword-spacing': ['warn', { before: true, after: true }],
      // 代码块 { 前留空格
      '@stylistic/space-before-blocks': ['warn', 'always'],
      // 单行代码块花括号内侧留空格：{ xxx(); } 而不是 {xxx();}
      '@stylistic/block-spacing': ['warn', 'always'],
      // 匿名函数/箭头函数的 ( 前留空格，具名函数不留（突出函数名）
      '@stylistic/space-before-function-paren': [
        'warn',
        { anonymous: 'always', named: 'never', asyncArrow: 'always' },
      ],
      // 函数名与 ( 之间不留空格
      '@stylistic/function-call-spacing': ['warn', 'never'],
      // 括号内侧不留空格：f(a) 而不是 f( a )
      '@stylistic/space-in-parens': ['warn', 'never'],
      // 二元运算符两侧留空格：a + b
      '@stylistic/space-infix-ops': 'warn',
      // 一元运算符紧贴操作数：!x、-y、typeof z
      '@stylistic/space-unary-ops': ['warn', { words: true, nonwords: false }],
      // 逗号前不留、后留空格：f(a, b)
      '@stylistic/comma-spacing': ['warn', { before: false, after: true }],
      // 分号前不留、后留空格
      '@stylistic/semi-spacing': ['warn', { before: false, after: true }],
      // 对象字面量冒号前不留、后留空格：{ a: 1 }
      '@stylistic/key-spacing': ['warn', { beforeColon: false, afterColon: true }],
      // switch 的 case 冒号后留空格
      '@stylistic/switch-colon-spacing': ['warn', { before: false, after: true }],
      // 箭头 => 两侧留空格：(a) => b
      '@stylistic/arrow-spacing': ['warn', { before: true, after: true }],
    },
  },
];
