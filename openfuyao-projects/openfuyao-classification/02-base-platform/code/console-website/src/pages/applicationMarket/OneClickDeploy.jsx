/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import '@/styles/applicationMarket/index.less';
import { yamlTojson } from '@/tools/utils';
import { Button, message } from 'antd';
import { useState, useRef, useEffect, useStore } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import copy from 'copy-to-clipboard';
import {
  createPod,
  createDeployment,
  createDaemonSet,
  createCronJob,
  createStatefulSet,
  addClusterRoleYamlData,
  addServiceAccountsYamlData,
  addRoleBindingsYamlData,
  createJob,
  createConfigMaps,
  createSecret,
  createResourceQuota,
  createLimitRange,
  addCustomResourceDefinitionYamlData,
  addServiceYamlData,
  addIngressYamlData,
  addNamespaceYamlData,
  addResourceInstanceYamlData,
  getCustomResourceDefinitionList,
  addPvYamlData,
  addScYamlData,
  addPvcYamlData,
  addClusterRoleBindingYamlData,
  addRoleYamlData,
  createSpecifiedPriorityClasses,
} from '@/api/containerApi';
import { exportYamlOutPut, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { containerRouterPrefix } from '@/constant.js';
import {
  ExportOutlined,
  CopyOutlined,
} from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import '@/styles/pages/nodeManage.less';
import { useHistory } from 'inula-router';
import { ResponseCode } from '@/common/constants';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function OneClickDeploy() {
  const history = useHistory();
  const [info, setInfo] = useState('');
  const [deployLoading, setDeployLoading] = useButtonLoading();
  const childCodeMirrorRef = useRef(null);
  const countRef = useRef(true);
  const [deployYaml, setDeployYaml] = useState('');
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const { t } = useTranslation(['appMarket', 'common']);

  const showCreateSuccess = (resource) => {
    messageApi.success(t('oneClickDeploy.createSuccess', { resource }));
  };

  const exportYaml = () => {
    exportYamlOutPut('Values', deployYaml);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  const handleCopyYaml = () => {
    copy(deployYaml);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };
  const parseYamlTimerRef = useRef(null);
  useEffect(() => () => clearTimeout(parseYamlTimerRef.current), []);
  const handleChangeDeployYaml = (e) => {
    setDeployYaml(e);
    clearTimeout(parseYamlTimerRef.current);
    parseYamlTimerRef.current = setTimeout(() => {
      if (!e) {
        setInfo([]);
        return;
      }
      let yaml;
      if (e.split('---').map((str) => str.trim())[0] === '') {
        yaml = e.split('---').map((str) => str.trim()).slice(1);
      } else {
        yaml = e.split('---').map((str) => str.trim());
      }
      try {
        setInfo(yaml.map((el) => yamlTojson(el)));
      } catch {
        setInfo([]);
      }
    }, 300);
  };

  const createCustomResource = async (arr) => {
    const apiVersion = arr.apiVersion || '';
    if (!apiVersion.includes('/')) {
      throw new Error(`Unsupported kind: ${arr.kind}`);
    }
    const sep = apiVersion.lastIndexOf('/');
    const group = apiVersion.slice(0, sep);
    const version = apiVersion.slice(sep + 1);

    const res = await getCustomResourceDefinitionList({
      customResourceDefinitionPage: 1,
      customResourceDefinitionPageSize: 10000,
      customResourceDefinitionContinue: '',
    });
    const crd = res.data?.items?.find(
      (item) => item.spec.group === group && item.spec.names.kind === arr.kind,
    );
    if (!crd) {
      throw new Error(`Unsupported kind: ${arr.kind}`);
    }

    const plural = crd.spec.names.plural;
    const namespace = crd.spec.scope === 'Namespaced'
      ? (arr.metadata?.namespace || 'default')
      : '';

    try {
      const createRes = await addResourceInstanceYamlData(
        { group, version, plural, namespace },
        arr,
      );
      if (createRes.status === ResponseCode.Created) {
        showCreateSuccess(arr.kind);
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        throw error;
      }
    }
  };

  const createResource = {
    Namespace: async (userData) => {
      try {
        const res = await addNamespaceYamlData(userData);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Namespace');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    ServiceAccount: async (userData) => {
      try {
        const res = await addServiceAccountsYamlData(userData.metadata.namespace || 'default', userData);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('ServiceAccount');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Service: async (arr) => {
      try {
        const res = await addServiceYamlData(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Service');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    ClusterRole: async (arr) => {
      try {
        const res = await addClusterRoleYamlData(arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('ClusterRole');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    ClusterRoleBinding: async (arr) => {
      try {
        const res = await addClusterRoleBindingYamlData(arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('ClusterRoleBinding');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Role: async (arr) => {
      try {
        const roleRes = await addRoleYamlData(arr.metadata.namespace || 'default', arr);
        if (roleRes.status === ResponseCode.Created) {
          showCreateSuccess('Role');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    RoleBinding: async (arr) => {
      try {
        const res = await addRoleBindingsYamlData(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('RoleBinding');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Pod: async (arr) => {
      try {
        const res = await createPod(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Pod');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Deployment: async (arr) => {
      try {
        const res = await createDeployment(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Deployment');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Job: async (arr) => {
      try {
        const res = await createJob(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Job');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    CronJob: async (arr) => {
      try {
        const res = await createCronJob(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('CronJob');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    ConfigMap: async (arr) => {
      try {
        const res = await createConfigMaps(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('ConfigMap');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Secret: async (arr) => {
      try {
        const res = await createSecret(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Secret');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    ResourceQuota: async (arr) => {
      try {
        const res = await createResourceQuota(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('ResourceQuota');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    LimitRange: async (arr) => {
      try {
        const res = await createLimitRange(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('LimitRange');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    CustomResourceDefinition: async (arr) => {
      try {
        const res = await addCustomResourceDefinitionYamlData(arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('CustomResourceDefinition');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    StatefulSet: async (arr) => {
      try {
        const res = await createStatefulSet(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('StatefulSet');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    Ingress: async (arr) => {
      try {
        const res = await addIngressYamlData(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('Ingress');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    DaemonSet: async (arr) => {
      try {
        const res = await createDaemonSet(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('DaemonSet');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    PersistentVolume: async (arr) => {
      try {
        const res = await addPvYamlData(arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('PersistentVolume');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    PersistentVolumeClaim: async (arr) => {
      try {
        const res = await addPvcYamlData(arr.metadata.namespace || 'default', arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('PersistentVolumeClaim');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    StorageClass: async (arr) => {
      try {
        const res = await addScYamlData(arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('StorageClass');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    PriorityClass: async (arr) => {
      try {
        const res = await createSpecifiedPriorityClasses(arr);
        if (res.status === ResponseCode.Created) {
          showCreateSuccess('PriorityClass');
        }
      } catch (error) {
        if (error?.response?.status === ResponseCode.Forbidden) {
          countRef.current = false;
          forbiddenMsg(messageApi, error);
        } else {
          throw error;
        }
      }
    },
    /**
     * 内置 kind 未命中时走 CR 创建：按 apiVersion + kind 匹配已安装 CRD，命中则创建实例，否则报不支持。
     */
    default: createCustomResource,
  };

  const createResources = async (infoArr) => {
    for (const config of infoArr) {
      await (createResource[config.kind] || createResource.default)(config);
    }
  };

  const handleDeploy = async () => {
    if (!Array.isArray(info) || !info.length) {
      messageApi.warning(t('oneClickDeploy.emptyYaml'));
      return;
    }
    countRef.current = true;
    setDeployLoading(true);
    try {
      await createResources(info);
      if (countRef.current) {
        await showSuccessThen(messageApi, t('oneClickDeploy.allCreateSuccess'), () => {
          history.push(`/${containerRouterPrefix}/workload/pod`);
        });
      }
    } catch (error) {
      countRef.current = false;
      messageApi.error(t('oneClickDeploy.createFailed', {
        message: error.response?.data?.message || error.message,
        interpolation: { escapeValue: false },
      }));
    } finally {
      setDeployLoading(false);
    }
  };

  const handleReset = () => {
    childCodeMirrorRef.current.resetCodeEditor();
  };

  const handleCancel = () => {
    history.goBack();
  };

  return (
    <div className={themeStore.$s.theme === 'light' ? '' : 'dark_box'}>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className='application_breadcrumb'>
        <BreadCrumbCom items={[{ title: t('oneClickDeploy.pageTitle'), disabled: true }]} />
      </div>
      <div className='one_deploy_container'>
        <div className='one_deploy_box normal_container_height'>
          <div className='yaml_card' style={{ padding: '0 0 32px' }}>
            <div className='yaml_flex_box'>
              <h3>YAML</h3>
              <div className='yaml_tools'>
                <div className='tool_word_group oneClickDeployExport' onClick={exportYaml}>
                  <ExportOutlined className='common_antd_icon primary_color' />
                  <span>{t('oneClickDeploy.export')}</span>
                </div>
                <div className='tool_word_group oneClickDeployCopy' onClick={handleCopyYaml}>
                  <CopyOutlined className='common_antd_icon primary_color' />
                  <span>{t('oneClickDeploy.copy')}</span>
                </div>
              </div>
            </div>
          </div>
          <CodeMirrorEditor
            changeYaml={handleChangeDeployYaml}
            ref={childCodeMirrorRef}
          />
          <div className='btn_container' style={{ padding: '32px 0' }}>
            <Button className='cancel_btn' onClick={handleCancel}>
              {t('actions.cancel', { ns: 'common' })}
            </Button>
            <Button className='cancel_btn' onClick={handleReset}>
              {t('actions.reset', { ns: 'common' })}
            </Button>
            <Button className='primary_btn' loading={deployLoading} onClick={handleDeploy}>
              {t('oneClickDeploy.deploy')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
