import { test, expect } from '@playwright/test';
import { chromium } from 'playwright'; 

// 连接到已打开的浏览器实例
async function connectToExistingBrowser() {
  try {
    const browser = await chromium.connectOverCDP('http://localhost:9222');
    const defaultContext = browser.contexts()[0];
    
    // 获取所有页面
    const pages = defaultContext.pages();
    
    if (pages.length === 0) {
      throw new Error('没有找到打开的页面');
    }
    
    console.log(`🔢 找到 ${pages.length} 个页面:`);
    pages.forEach((p, i) => {
      console.log(`  ${i + 1}. ${p.url()}`);
    });
    
    // 查找包含目标URL的页面
    const targetKeyword = '192.168.200.8';
    let targetPage = null;
    
    for (const page of pages) {
      const url = page.url();
      if (url.includes(targetKeyword)) {
        targetPage = page;
        console.log(`✅ 找到目标页面: ${url}`);
        break;
      }
    }
    
    // 如果没有找到目标页面，直接退出
    if (!targetPage) {
      console.error(`❌ 未找到包含 ${targetKeyword} 的页面`);
      console.error('💡 请确保已打开目标页面');
      await browser.close();
      process.exit(1); // 退出程序
    }
    
    console.log(`✅ 成功连接到现有浏览器`);
    console.log(`📄 使用页面: ${await targetPage.url()}`);
    
    return { browser, page: targetPage };
  } catch (error) {
    console.error('❌ 无法连接到现有浏览器:', error.message);
    console.log('💡 请确保已启动Chrome并开启调试端口：');
    console.log('   chrome.exe --remote-debugging-port=9222');
    throw error;
  }
}

// 填写IP字段的辅助函数
async function fillIPField(page, tid, ipSegments) {
  for (let i = 0; i < ipSegments.length; i++) {
    await page.locator(`[tid="${tid}"]`).locator(`input.ti_input_ip_octet:nth-of-type(${i + 1})`).fill(ipSegments[i]);
  }
}

// 配置虚拟机网络的完整流程
async function configureVMNetwork(page, treeId, ipEnd) {
  console.log(`正在配置虚拟机 ${treeId}，IP: 192.168.200.${ipEnd}`);
  // 触发页面刷新
  //await page.reload();
  //await page.waitForTimeout(3000);
  // 点击虚拟机树节点
  await reliableClick(page, `#ztree_${treeId}_a`, {
  maxRetries: 3,
  waitBeforeClick: 1500,
  scrollToCenter: true
  });
  
  // 进入配置页面
  await page.getByRole('link', { name: '配置' }).click();
  await page.locator('ti-leftmenu-item').filter({ hasText: '硬件' }).click();
  await page.getByText('网卡1个').click();
  
  // 打开更多菜单
  await page.getByRole('cell', { name: '修改端口组 更多' }).locator('#opsBtn_menu_toggle').click();
  await page.locator('#opsBtn_menu_menulist0_editNetwork').click();
  
  // 启用网络配置
  await page.locator('#enableNetConfig_awrapper').click();
  //await page.waitForTimeout(1000); // 等待页面加载
  await page.getByText('网络配置1：IPv4').click();
  await page.waitForTimeout(1000);
  await page.locator('#autoGetIP_dominator_btn').click();
  await page.locator('section').filter({ hasText: '使用静态IP地址' }).click();
  
  // 填写网络配置
  await fillIPField(page, 'ipAddr', ['192', '168', '200', ipEnd.toString()]);
  await fillIPField(page, 'ipMask', ['255', '255', '255', '0']);
  await fillIPField(page, 'ipGateway', ['192', '168', '200', '1']);
  await fillIPField(page, 'ipv4DNS1', ['1', '1', '1', '1']);
  
  // 保存配置
  await page.getByRole('button', { name: '确定' }).click();
  
  // 返回概要页面
  await page.getByRole('link', { name: '概要' }).click();
}

async function reliableClick(page, selector, options = {}) {
  const {
    maxRetries = 3,
    waitBeforeClick = 1000,
    waitAfterClick = 500,
    scrollToCenter = true,
    useForceOnLastRetry = true
  } = options;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`尝试点击 ${selector} (第${attempt}次)`);
      
      const preClickUrl = await page.url();//记录之前的url

      // 1. 等待元素存在
      await page.waitForSelector(selector, { 
        state: 'attached', 
        timeout: 5000 
      });
      
      // 2. 滚动到视图中心
      if (scrollToCenter) {
        await page.evaluate((sel) => {
          const element = document.querySelector(sel);
          if (element) {
            // 先滚动到顶部，再滚动到中心，确保完全可见
            element.scrollIntoView(true);
            element.scrollIntoView({ 
              behavior: 'smooth', 
              block: 'center',
              inline: 'center'
            });
          }
        }, selector);
      }
      
      // 3. 等待页面稳定
      await page.waitForTimeout(waitBeforeClick);
      
      // 4. 检查元素状态（非强制模式下）
      const isForce = useForceOnLastRetry && attempt === maxRetries;
      
      if (!isForce) {
        const element = page.locator(selector);
        const isVisible = await element.isVisible();
        const isEnabled = await element.isEnabled();
        
        if (!isVisible || !isEnabled) {
          throw new Error(`元素不可点击: visible=${isVisible}, enabled=${isEnabled}`);
        }
      }
      
      // 5. 执行点击
      await page.click(selector, { 
        timeout: 3000,
        force: isForce,
        // 可选的：模拟真实点击
        delay: 100
      });
      
      // 6. 等待点击后的响应
      await page.waitForTimeout(waitAfterClick);
      
      // 7. 验证点击是否成功
      const clickedSuccess = await verifyClickSuccess(page, selector, preClickUrl);
      
      if (clickedSuccess) {
        console.log(`✅ 成功点击 ${selector}`);
        return true;
      } else {
        throw new Error('点击后页面状态未变化');
      }
      
    } catch (error) {
      console.warn(`点击失败 (尝试 ${attempt}/${maxRetries}): ${error.message}`);
      
      if (attempt >= maxRetries) {
        console.error(`❌ 无法点击元素 ${selector} (已达到最大尝试次数)`);
        throw new Error(`无法点击元素 ${selector}: ${error.message}`);
      }

      // 重试前等待
      await page.waitForTimeout(2000);
    }
  }
}

// 验证点击是否成功的辅助函数
async function verifyClickSuccess(page, selector, preClickUrl) {
  try {
  // 获取点击后的URL
  const postClickUrl = await page.url();
  
  console.log('验证点击结果:');
  console.log('  点击前URL:', preClickUrl);
  console.log('  点击后URL:', postClickUrl);
  
  // 两个条件都必须满足
  if (preClickUrl !== postClickUrl) {
    const preUrlObj = new URL(preClickUrl);
    const postUrlObj = new URL(postClickUrl);
    
    if (preUrlObj.hash !== postUrlObj.hash) {
      console.log('  ✅ URL和hash都发生了变化');
      return true;
    } else {
      console.log('  ❌ URL不同但hash相同');
      return false;
    }
  }
  
  console.log('  ❌ URL未发生变化');
  return false;
  
  } catch (error) {
    console.error('验证点击结果时出错:', error.message);
    return false;
  }
}

// 重启虚拟机
async function rebootVM(page, treeId) {
  await page.locator(`#ztree_${treeId}_a`).click({ button: 'right' });
  await page.locator('#powerMenu').getByText('电源', { exact: true }).click();
  await page.locator('#rebootMenu').getByText('重启').click();
  await page.waitForTimeout(5000);
  await page.getByRole('button', { name: '确定' }).click();
  await page.waitForTimeout(2000);
}


let globalBrowser = null;
let globalPage = null;
test.describe('批量配置虚拟机网络', () => {
   // 所有测试前：连接到现有浏览器
  test.beforeAll(async () => {
    console.log('🔗 连接到现有浏览器...');
    const connection = await connectToExistingBrowser();
    globalBrowser = connection.browser;
    globalPage = connection.page;
    
    // 打印当前页面信息
    const currentUrl = await globalPage.url();
    console.log(`🌐 当前页面URL: ${currentUrl}`);
    
    // 等待页面完全加载
    await globalPage.waitForLoadState('networkidle');
  });
  
  // 所有测试后：只断开连接，不关闭浏览器
  test.afterAll(async () => {
    if (globalBrowser) {
      console.log('🔌 断开浏览器连接（保持浏览器打开）');
      await globalBrowser.close();
    }
  });
  
  // 每个测试前：确保页面就绪
  test.beforeEach(async () => {
    // 确保页面是激活状态
    await globalPage.bringToFront();
    
    // 添加一些基础检查
    const url = await globalPage.url();
    console.log(`📄 测试开始，当前页面: ${url}`);
    
    // 如果页面空白，刷新一下
    if (url === 'about:blank' || url.includes('chrome://')) {
      console.log('🔄 刷新页面...');
      await globalPage.reload();
      await globalPage.waitForLoadState('networkidle');
    }
  });

  test('批量配置n台虚拟机', async () => {
    const startTreeId = 327;
    const startIpEnd = 156;
    const vmCount = 9;
    
    // 超时配置
    const normalTimeoutPerVM = 25000; // 正常单次循环超时：25秒
    const maxTimeoutPerVM = 35000;   // 最大单次循环超时：35s（防止卡死）
    const bufferTime = 30000;         // 总缓冲时间：30秒
    
    // 计算总超时时间
    const totalTimeout = vmCount * normalTimeoutPerVM + bufferTime;
    test.setTimeout(totalTimeout);
    
    console.log(`📊 超时配置:`);
    console.log(`  单次循环最大超时: ${maxTimeoutPerVM/1000}秒`);
    console.log(`  总超时时间: ${totalTimeout/1000}秒`);
    console.log(`  虚拟机数量: ${vmCount}台`);
    
    for (let i = 0; i < vmCount; i++) {
        const currentTreeId = startTreeId + i;
        const currentIpEnd = startIpEnd + i;
        
        console.log(`\n🚀 开始配置第 ${i + 1}/${vmCount} 台虚拟机 (ID: ${currentTreeId})`);
        
        const loopStartTime = Date.now();
        
        try {
        // 为单次循环设置超时控制 - 超时直接抛出错误中断测试
        await Promise.race([
            // 正常的配置流程
            (async () => {
            // 配置网络
            await configureVMNetwork(globalPage, currentTreeId, currentIpEnd);
            
            // 重启虚拟机
            await rebootVM(globalPage, currentTreeId);
            })(),
            
            // 超时控制 - 超时直接抛出错误
            new Promise((_, reject) => 
            setTimeout(() => {
                const elapsed = (Date.now() - loopStartTime) / 1000;
                reject(new Error(`❌ 第 ${i + 1} 台虚拟机配置超时（已运行${elapsed.toFixed(1)}秒，超过${maxTimeoutPerVM/1000}秒限制）`));
            }, maxTimeoutPerVM)
            )
        ]);
        
        } catch (error) {
            console.error(`\n🔴 测试中断！`);
            console.error(`   失败位置: 第 ${i + 1} 台虚拟机`);
            console.error(`   失败原因: ${error.message}`);
            console.error(`   已配置完成: ${i} 台`);
            console.error(`   未完成: ${vmCount - i} 台`);
            
            // 可以直接抛出错误，让测试失败
            throw error;
        }
        
        const loopElapsedTime = Date.now() - loopStartTime;
        const remainingVMs = vmCount - (i + 1);
        const estimatedRemainingTime = remainingVMs * (loopElapsedTime / (i + 1));
        
        console.log(`✅ 第 ${i + 1} 台虚拟机配置完成`);
        console.log(`   本次耗时: ${(loopElapsedTime/1000).toFixed(1)}秒`);
        
        // 如果本次接近超时，给出警告
        if (loopElapsedTime > maxTimeoutPerVM * 0.8) {
            console.warn(`⚠️  警告：本次耗时接近超时限制（${(loopElapsedTime/1000).toFixed(1)}/${maxTimeoutPerVM/1000}秒）`);
        }
        
        // 如果还有剩余虚拟机，显示预估时间
        if (remainingVMs > 0) {
            console.log(`  剩余 ${remainingVMs} 台，预估还需 ${(estimatedRemainingTime/1000).toFixed(1)} 秒`);
        }
    }
    
    console.log(`\n🎉 所有 ${vmCount} 台虚拟机配置完成！`);
 });
});