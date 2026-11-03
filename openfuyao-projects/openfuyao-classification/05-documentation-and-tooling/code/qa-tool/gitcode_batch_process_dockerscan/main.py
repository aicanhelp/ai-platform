# -*- coding: utf-8 -*-
import time
import os
import logging
from selenium import webdriver
from selenium.webdriver.edge.service import Service  
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------- 手动配置参数 --------------------------
# 镜像仓配置
HARBOR_BASE_URL = "镜像仓地址"  
HARBOR_USER = "镜像仓用户名"              
HARBOR_PWD = "镜像仓密码"            

# 浏览器配置
WEB_URL = "浏览器url"
WEB_USER = ""  # 无需登录则留空
WEB_PWD = ""   # 无需登录则留空

# 任务配置（根据需要处理的任务进行修改）
TARGET_REGION = ""                 # 目标区域
SCAN_ENGINES = []  # 扫描引擎
NEED_TEST_CONNECTION = True          # 是否勾选连接测试

IMAGES_LIST = "../images.txt"  # 镜像列表路径
EDGE_DRIVER_PATH = "../AppData/Local/Programs/Python/Python39/msedgedriver.exe"  # 驱动路径


# -------------------------- 核心工具函数 --------------------------
def init_edge_driver():
    """初始化浏览器（修复SSL握手+优化加载）"""
    edge_options = webdriver.EdgeOptions()
    edge_options.add_argument("--start-maximized")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--ignore-certificate-errors")
    edge_options.add_argument("--ignore-ssl-errors")
    
    prefs = {"pageLoadStrategy": "eager"}
    edge_options.add_experimental_option("prefs", prefs)
    
    try:
        service = Service(executable_path=EDGE_DRIVER_PATH)
        driver = webdriver.Edge(service=service, options=edge_options)
        driver.implicitly_wait(30)
        driver.set_page_load_timeout(60)
        return driver
    except Exception as e:
        logging.warning(f"驱动初始化失败：{str(e)}（检查驱动路径）")
        return None


def login_web(driver):
    """登录浏览器"""
    try:
        logging.info(f"🔄 访问浏览器：{WEB_URL}")
        driver.get(WEB_URL)
        time.sleep(5)
        save_page_diagnosis(driver, "web_initial") 
        
        if WEB_USER.strip() and WEB_PWD.strip():
            user_input = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//input[@name='username' or @id='username']"))
            )
            pwd_input = driver.find_element(By.XPATH, "//input[@name='password' or @id='password']")
            
            user_input.clear()
            user_input.send_keys(WEB_USER)
            pwd_input.clear()
            pwd_input.send_keys(WEB_PWD)
            
            login_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' or contains(text(), '登录')]"))
            )
            login_btn.click()
            time.sleep(5)
            save_page_diagnosis(driver, "web_after_login")
        
        if "404" in driver.title or "500" in driver.title:
            logging.info(f"页面异常：{driver.title}（可能URL错误）") 

        logging.info("✅ 浏览器访问成功")
        return True
    except Exception as e:
        logging.warning(f"❌ 访问浏览器失败：{str(e)}")
        save_page_diagnosis(driver, "web_error")
        return False


def click_add_button(driver):
    """步骤1：点击添加按钮"""
    add_methods = [
        (By.ID, "taskServiceList_btn_add_button"),
        (By.XPATH, "//button[.//span[@class='button-content' and text()='添加']]")
        ]
    for by, selector in add_methods:
        try:
            add_btn = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((by, selector)))
            add_btn.click()
            time.sleep(3)
            logging.info("✅ 已点击'添加'按钮")
            return
        except Exception as e:
            logging.warning(
                f"❌ 使用选择器 {by}:{selector} 点击'添加'按钮失败: {str(e)}",
                exc_info=False  # 如需详细堆栈信息可改为 True
            )
            continue  

    driver.save_screenshot("add_button_not_found.png")
    logging.info("未找到'添加'按钮（见截图）")


def fill_task_name(driver, image_name):
    """步骤2：填写任务名称（在默认名称后添加_镜像名）"""
    processed_image = image_name.replace('/', '_').replace(':', '_')
    task_input = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.ID, "taskName"))  
    )
    default_name = task_input.get_attribute("value") or ""  
    
    if default_name:
        task_name = f"{default_name}_{processed_image}"
    else:
        task_name = processed_image
    
    task_input.clear()
    task_input.send_keys(task_name)
    logging.info(f"✅ 已填写任务名称：{task_name}")
    return task_name


def select_target_region(driver, target_region):
    """步骤3：选择目标区域"""
    target_label = WebDriverWait(driver, 15).until(
    EC.element_to_be_clickable(
        (By.XPATH, 
         f"//label[.//span[@class='devui-radio-label' and text()='{target_region}']]")
    )
)
    if "active" not in target_label.get_attribute("class"):
        target_label.click()
        logging.info(f"✅ 已勾选'{target_region}'")
    else:
        logging.info(f"ℹ️ '{target_region}'已选中")
    
    try:
        test_label = driver.find_element(
            By.XPATH, "//label[.//span[@class='devui-radio-label' and text()='**']]"   # "**"根据自己要处理的浏览器界面进行修改
        )
        if "active" in test_label.get_attribute("class"):
            test_label.click()
            logging.info("✅ 已取消'**'")            # "**"根据自己要处理的浏览器界面进行修改
    except Exception as e:
        logging.info(f"ℹ️ 处理**异常：{str(e)}")      # "**"根据自己要处理的浏览器界面进行修改


def check_scan_engines(driver, engines):
    """步骤4：勾选扫描引擎"""
    for engine in engines:
        engine_label = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable(
            (
                By.XPATH, 
                f"//label[.//span[contains(@class, 'devui-checkbox-material')]"
                f"/following-sibling::text()[normalize-space()='{engine}']]"
            )
        )
    )
        checkbox_div = engine_label.find_element(By.XPATH, "./parent::div")
        if "unchecked" in checkbox_div.get_attribute("class"):
            engine_label.click()
            logging.info(f"✅ 已勾选引擎：{engine}")
        else:
            logging.info(f"ℹ️ 引擎'{engine}'已选中")


def check_connection_test(driver):
    """步骤5：勾选连接测试（按需）"""
    test_label = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, 
            "//div[contains(text(), '连接测试')]/ancestor::d-form-item//label[.//input[@type='checkbox']]"))
    )
    checkbox_div = test_label.find_element(By.XPATH, "./parent::div")
    if "unchecked" in checkbox_div.get_attribute("class"):
        test_label.click()
        logging.info("✅ 已勾选'连接测试'")
    else:
        logging.info("ℹ️ '连接测试'已选中")


def fill_image_info(driver, full_url, username, password):
    """步骤6：填写镜像地址和认证信息（使用明确ID定位）"""
    expected_prefix = f"{HARBOR_BASE_URL}/"
    if not full_url.startswith(expected_prefix):
        logging.warning(f"镜像地址格式错误！应为'{expected_prefix}<镜像名>'，实际为'{full_url}'")
    
    url_input = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.ID, "dockerAddress"))
    )
    url_input.clear()
    url_input.send_keys(full_url)
    image_name = full_url[len(expected_prefix):]
    logging.info(f"✅ 已填写镜像地址：{HARBOR_BASE_URL}/<{image_name}>")
    
    user_input = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.ID, "dockerUser"))
    )
    user_input.clear()
    user_input.send_keys(username)
    logging.info("✅ 已填写仓库用户名")
    
    pwd_input = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.ID, "common_password"))
    )
    pwd_input.clear()
    pwd_input.send_keys(password)
    logging.info("✅ 已填写仓库密码")


def save_and_run_task(driver, task_name):
    """步骤7：保存并运行任务"""
    save_btn = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.ID, "dockerCreateEdit_button"))
    )
    btn_text = save_btn.find_element(By.XPATH, ".//span[@class='button-content']").text.strip()
    logging.info(f"ℹ️ 点击按钮：{btn_text}")
    save_btn.click()
    
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(text(), '成功') and not(contains(text(), '失败'))]"))
    )
    logging.info(f"✅ 任务创建成功：{task_name}")


def reset_after_task(driver):
    """任务完成后重置页面状态"""
    try:
        close_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '关闭') or contains(text(), '确定')]"))
        )
        close_btn.click()
        time.sleep(2)
    except Exception as e:
        driver.refresh()
        time.sleep(3)
        logging.info("🔄 已重置页面状态")


def read_images_list():
    if not os.path.exists(IMAGES_LIST):
        logging.warning(f"镜像列表文件不存在：{IMAGES_LIST}")
    
    with open(IMAGES_LIST, "r", encoding="utf-8") as f:
        images = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                images.append({
                    "original": line,
                    "full_url": f"{HARBOR_BASE_URL}/{line}"
                })
    
    if not images:
        logging.warning("images.txt中无有效镜像")
    
    logging.info(f"📋 读取到{len(images)}个镜像：")
    for img in images:
        logging.info(f"  - {img['original']}")
    return images


def save_page_diagnosis(driver, prefix):
    """保存页面诊断信息（截图+源码）"""
    try:
        driver.save_screenshot(f"{prefix}_screenshot.png")
        with open(f"{prefix}_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logging.info(f"📸 诊断文件已保存：{prefix}_*")
    except Exception as e:
        logging.info(f"⚠️ 保存诊断信息失败：{str(e)}")


def create_single_scan_task(driver, img_info):
    """创建单个扫描任务（完整流程）"""
    task_name = None
    try:
        click_add_button(driver)
        task_name = fill_task_name(driver, img_info["original"])
        select_target_region(driver, TARGET_REGION)
        check_scan_engines(driver, SCAN_ENGINES)
        
        if NEED_TEST_CONNECTION:
            check_connection_test(driver)
        else:
            logging.info("ℹ️ 跳过连接测试（配置为不勾选）")
        
        fill_image_info(driver, img_info["full_url"], HARBOR_USER, HARBOR_PWD)
        
        save_and_run_task(driver, task_name)
        reset_after_task(driver)
        return True
        
    except Exception as e:
        logging.info(f"❌ 任务创建失败：{str(e)}")
        save_page_diagnosis(driver, f"task_error_{task_name or img_info['original'][:10]}")
        try:
            driver.find_element(By.XPATH, "//button[contains(text(), '取消')]").click()
            time.sleep(2)
        except Exception as close_err:
            logging.warning(f"⚠️ 尝试关闭弹窗失败：{str(close_err)}")
        return False


def main():
    """主函数：批量处理镜像扫描任务"""
    # 1. 读取镜像列表（仅一次）
    try:
        images = read_images_list()
    except Exception as e:
        logging.info(f"❌ 初始化失败：{str(e)}")
        input("按Enter退出...")
        return
    
    # 2. 初始化浏览器
    try:
        driver = init_edge_driver()
    except Exception as e:
        logging.info(f"❌ 浏览器启动失败：{str(e)}")
        input("按Enter退出...")
        return
    
    try:
        # 3. 登录浏览器界面
        if not login_web(driver):
            input("按Enter退出...")
            return
        
        # 4. 批量创建任务
        success_count = 0
        fail_count = 0
        
        for idx, img in enumerate(images, 1):
            logging.info(f"\n{'='*50}")
            logging.info(f"处理第{idx}/{len(images)}个镜像：{img['original']}")
            logging.info(f"{'='*50}")
            
            if create_single_scan_task(driver, img):
                success_count += 1
            else:
                fail_count += 1
        
        # 5. 输出统计结果
        logging.info(f"\n{'='*60}")
        logging.info(f"批量处理完成 | 总计：{len(images)} | 成功：{success_count} | 失败：{fail_count}")
        logging.info(f"{'='*60}")
    
    except Exception as e:
        logging.info(f"\n❌ 程序异常终止：{str(e)}")
    finally:
        input("\n按Enter关闭浏览器...")
        driver.quit()


if __name__ == "__main__":
    main()
    