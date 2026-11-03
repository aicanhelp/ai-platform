/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { ResponseCode } from '../common/constants';
import { useHistory, Link } from 'inula-router';
import { useEffect, useState, useCallback, createRef } from 'openinula';
import { Form, Table, ConfigProvider, Space, Input, Button, message, Popover, Spin, Flex, Modal } from 'antd';
import '@/styles/cluster.less';
import { MoreOutlined } from '@ant-design/icons';
import zhCN from 'antd/es/locale/zh_CN';
import { sorterFirstAlphabe, getkeysList, memorySorter, cpuSorter } from '@/utils/common';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { getClusters, deleteCluster, editClusterLabel, getCertificate } from '@/api/multiclusterApi';
import hostImg from '@/assets/images/host.png';
import AnnotationModal from '@/components/AnnotationModal';
import '@/styles/dark.less';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { jsonToYaml } from '@/utils/common';
import { useIntl } from 'inula-intl';

export default function MultiCluster() {
  const intl = useIntl();
  const history = useHistory();
  const childCodeMirrorRef = createRef(null);
  const [clusterForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [memberPopOpen, setMemberPopOpen] = useState(''); // 悬浮框是否展示
  const [clusterPage, setClusterPage] = useState(1); // 分页
  const [hostInfo, setHostInfo] = useState();
  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [clusterDelModalOpen, setClusterDelModalOpen] = useState(false); // 删除对话框展示
  const [clusterDelName, setClusterDelName] = useState(''); // 删除的名称
  const [isClusterDelCheck, setIsClusterDelCheck] = useState(false); // 是否选中
  const [clusterList, setClusterList] = useState();
  const [delLoading, setDelLoading] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [labels, setLabels] = useState([]);
  const [oldLabels, setOldLabels] = useState([]);
  const [editLabelName, setEditLabelName] = useState();
  const [isShowHost, setIsShowHost] = useState(true);
  const [isShowCertificate, setIsShowCertificate] = useState(false);
  const [certificateInfo, setCertificateInfo] = useState();
  const [isUpdate, setIsUpdate] = useState(false);
  // 列表项
  const clusterColumns = [
    {
      title: intl.formatMessage({ id: 'multicluster.clusterName' }),
      key: 'cluster_name',
      sorter: (a, b) => sorterFirstAlphabe(a.clustername, b.clustername),
      render: (_, record) => (
        <p className="table_cluser_click" onClick={() => goConsoleWebsite(record.clustername)}>
          {record.clustername}
        </p>
      ),
    },
    {
      title: intl.formatMessage({ id: 'multicluster.clusterLabels' }),
      key: 'ClusterLabel',
      render: (_, record) => (
        <Space style={{ display: 'flex', justifyContent: 'start', flexWrap: 'wrap' }}>
          {record.clusterlabels ? (
            record.clusterlabels.map((item) => (
              <div className="table_label">
                <p className="table_label_key">{item.split(':')[0]}</p>
                <p className="table_label_value">{item.split(':')[1]}</p>
              </div>
            ))
          ) : (
            <div className="table_label">{intl.formatMessage({ id: 'multicluster.noLabels' })}</div>
          )}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'multicluster.cpuRequestTotal' }),
      key: 'CpuNum',
      sorter: (a, b) => cpuSorter(a, b),
      render: (_, record) =>
        `${record.clusterresources.allocatedcpu}` +
        `Core` +
        ' / ' +
        `${record.clusterresources.allocatablecpu}` +
        `Core`,
    },
    {
      title: intl.formatMessage({ id: 'multicluster.memoryRequestTotal' }),
      key: 'MemoryNum',
      sorter: (a, b) => memorySorter(a, b),
      render: (_, record) =>
        `${record.clusterresources.allocatedmem}` + ' / ' + `${record.clusterresources.allocatablemem}`,
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      key: 'handle',
      fixed: 'right',
      width: 120,
      render: (_, record) => (
        <Space>
          <Popover
            placement="bottom"
            content={
              <div className="pop_modal" style={{ display: 'flex', flexDirection: 'column' }}>
                <Button type="link" onClick={() => handleCertificate(record.clustername)}>
                  {intl.formatMessage({ id: 'multicluster.viewCredential' })}
                </Button>
                <Button type="link" onClick={() => editLabelLabel(record)}>
                  {intl.formatMessage({ id: 'multicluster.editLabel' })}
                </Button>
                <Button type="link" onClick={() => handleDeleteMember(record)}>
                  {intl.formatMessage({ id: 'multicluster.removeManagement' })}
                </Button>
              </div>
            }
            trigger="click"
            open={memberPopOpen === record.clustername}
            onOpenChange={(e) => handleMemberOpenChange(e, record)}
          >
            <MoreOutlined style={{ color: '#346ac4', fontSize: '20px' }} />
          </Popover>
        </Space>
      ),
    },
  ];
  const goConsoleWebsite = (name) => {
    sessionStorage.setItem('cluster', name);
    window.location.href = `/container_platform/`;
  };
  const editLabelLabel = (data) => {
    setMemberPopOpen('');
    let labelList = [];
    if (data.clusterlabels) {
      data.clusterlabels.map((item) => {
        labelList.push({
          key: item.split(':')[0],
          value: item.split(':')[1],
        });
      });
    } else {
      labelList = [{ key: '', value: '' }];
    }
    setEditLabelName(data.clustername);
    setLabels(labelList);
    setOldLabels(labelList);
    setIsLabelModalOpen(true);
  };

  // 成员集群修改删除
  const handleMemberOpenChange = (newOpen, record) => {
    if (newOpen) {
      setMemberPopOpen(record.clustername);
    } else {
      setMemberPopOpen('');
    }
  };
  const handleDeleteMember = async (record) => {
    setClusterDelName(record.clustername);
    setMemberPopOpen('');
    setClusterDelModalOpen(true); // 打开弹窗
  };
  const handleDelClusterCancel = () => {
    setClusterDelModalOpen(false);
    setClusterDelName('');
  };
  const handleCertificate = async (data) => {
    setMemberPopOpen('');
    try {
      const res = await getCertificate(data);
      if (res.status === ResponseCode.OK) {
        setIsUpdate(true);
        setCertificateInfo(jsonToYaml(JSON.stringify(res.data)).replaceAll('>-', ''));
        setIsShowCertificate(true);
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(intl.formatMessage({ id: 'multicluster.noPermission' }));
      } else {
        messageApi.error(
          intl.formatMessage({ id: 'multicluster.viewCredentialFailed' }, { msg: error.response.data.Message }),
        );
      }
    }
  };
  const handleCancel = () => {
    setIsUpdate(false);
    setIsShowCertificate(false);
  };
  const handleClustertConfirm = async () => {
    setClusterDelModalOpen(false);
    setDelLoading(true);
    try {
      const res = await deleteCluster(clusterDelName);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'multicluster.removeClusterSuccess' }, { name: clusterDelName }));
        if (sessionStorage.getItem('cluster') === clusterDelName) {
          if (hostInfo?.clustername) {
            sessionStorage.setItem('cluster', hostInfo?.clustername);
          } else {
            sessionStorage.setItem(
              'cluster',
              clusterList.filter((item) => item.clustername !== clusterDelName)[0].clustername,
            );
          }
        }
        setDelLoading(false);
        setClusterDelModalOpen(false);
        setIsClusterDelCheck(false);
        setClusterDelName('');
        getClusterList();
      }
    } catch (e) {
      setDelLoading(false);
      messageApi.error(intl.formatMessage({ id: 'multicluster.removeClusterFailed' }, { name: clusterDelName }));
    }
  };
  const handleDeploymentCheckFn = (e) => {
    setIsClusterDelCheck(e.target.checked);
  };

  // 获取clusterList
  const getClusterList = useCallback(async (isChange = true) => {
    try {
      const res = await getClusters();
      if (res.status === ResponseCode.OK) {
        const lists = [];
        getkeysList(res.data.info).map((item) => {
          lists.push({
            clustername: res.data.info[item].clustername,
            clusterlabels: res.data.info[item].clusterlabels,
            clusterresources: res.data.info[item].clusterresources,
          });
        });
        if (
          lists.map((item) => {
            if (item.clustername === 'host') {
              item.type = item.clustername === 'host' ? 'host' : 'other';
            }
          })
        ) {
          setOriginalList([...lists]);
        }
        let info = {
          clustername: lists.filter((item) => item.type === 'host')[0]?.clustername,
          clusterresources: lists.filter((item) => item.type === 'host')[0]?.clusterresources,
          clusterlabels: lists.filter((item) => item.type === 'host')[0]?.clusterlabels,
          type: lists.filter((item) => item.type === 'host')[0]?.type,
        };
        setHostInfo(info);
        setIsShowHost(lists.filter((item) => item.clustername === 'host').length > 0);
        handleSearchClusterment(lists, isChange);
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'multicluster.getClusterListFailed' }));
    }
  }, []);

  // 检索
  const handleSearchClusterment = (totalData = originalList, isChange = true) => {
    const clusterFormName = clusterForm.getFieldValue('cluster_name');
    let temporyList = totalData;
    if (clusterFormName) {
      temporyList = temporyList.filter((item) =>
        item.clustername.toLowerCase().includes(clusterFormName.toLowerCase()),
      );
    }
    setClusterList(temporyList.filter((item) => item.type !== 'host'));
    isChange ? setClusterPage(1) : null;
  };

  // 标签成功回调
  const handleLabelOkConfigurationDetail = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      message.info(intl.formatMessage({ id: 'multicluster.labelsNotModified' }));
      setIsLabelModalOpen(false);
      return;
    }
    const keyArrDetailLabel = [];
    // 判断是否存在相同key
    data.map((item) => keyArrDetailLabel.push(item.key));
    if (keyArrDetailLabel.filter((item, index) => keyArrDetailLabel.indexOf(item) !== index).length) {
      message.error(intl.formatMessage({ id: 'multicluster.duplicateKey' }));
    } else {
      // 请求接口添加
      // 比较前后是否一致 返回处理后的标签
      let paramLabel = { clusterlabels: {} };
      data.map((item) => {
        paramLabel.clusterlabels[item.key] = item.value;
      });
      try {
        await editClusterLabel(editLabelName, paramLabel);
        message.success(intl.formatMessage({ id: 'multicluster.editLabelSuccess' }));
        setTimeout(() => {
          setIsLabelModalOpen(false);
          setEditLabelName('');
          getClusterList();
        }, 1000);
      } catch (error) {
        message.error(intl.formatMessage({ id: 'multicluster.editLabelFailed' }, { msg: error.response.data.Message }));
      }
    }
  };

  // 标签失败回调
  const handleLabelCancelDetail = () => {
    setIsLabelModalOpen(false);
    setEditLabelName('');
  };

  useEffect(() => {
    getClusterList();
  }, []);

  return (
    <div
      className="child_content munlti_cluster_index"
      style={{ background: localStorage.theme === 'dark' ? '##171a1f' : '#f7f7f7' }}
    >
      {contextHolder}
      <p className="title" style={{ display: isShowHost ? 'block' : 'none' }}>
        {intl.formatMessage({ id: 'multicluster.hostCluster' })}
      </p>
      <div className="host_cluster" style={{ display: isShowHost ? 'flex' : 'none' }}>
        <div className="host_cluster_left">
          <img className="host_cluster_left_img" src={hostImg} alt="" />
          <div className="host_cluster_left_info">
            <p className="host_cluster_left_name" onClick={() => goConsoleWebsite(hostInfo?.clustername)}>
              {hostInfo?.clustername}
            </p>
            <p className="host_cluster_left_title">{intl.formatMessage({ id: 'multicluster.clusterName' })}</p>
          </div>
          <div className="host_cluster_left_role">{intl.formatMessage({ id: 'multicluster.hostCluster' })}</div>
        </div>
        <div className="host_cluster_right">
          <div className="host_cluster_right_info">
            <p className="host_cluster_right_text">{hostInfo?.clusterresources?.allocatedcpu}Core</p>
            <p className="host_cluster_right_title">{intl.formatMessage({ id: 'multicluster.cpuRequest' })}</p>
          </div>
          <div className="host_cluster_right_info">
            <p className="host_cluster_right_text">{hostInfo?.clusterresources?.allocatablecpu}Core</p>
            <p className="host_cluster_right_title">{intl.formatMessage({ id: 'multicluster.cpuTotal' })}</p>
          </div>
          <div className="host_cluster_right_info">
            <p className="host_cluster_right_text">{hostInfo?.clusterresources?.allocatedmem}</p>
            <p className="host_cluster_right_title">{intl.formatMessage({ id: 'multicluster.memoryRequest' })}</p>
          </div>
          <div className="host_cluster_right_info">
            <p className="host_cluster_right_text">{hostInfo?.clusterresources?.allocatablemem}</p>
            <p className="host_cluster_right_title">{intl.formatMessage({ id: 'multicluster.memoryTotal' })}</p>
          </div>
          <div className="host_cluster_right_info" style={{ paddingTop: '16px' }}>
            <Popover
              placement="bottom"
              content={
                <div className="pop_modal" style={{ display: 'flex', flexDirection: 'column' }}>
                  <Button type="link" onClick={() => handleCertificate(hostInfo?.clustername)}>
                    {intl.formatMessage({ id: 'multicluster.viewCredential' })}
                  </Button>
                </div>
              }
              trigger="click"
              open={memberPopOpen === hostInfo?.clustername}
              onOpenChange={(e) => handleMemberOpenChange(e, hostInfo)}
            >
              <MoreOutlined style={{ color: '#346ac4', fontSize: '20px' }} />
            </Popover>
          </div>
        </div>
      </div>
      <p className="title memebr_management">
        {intl.formatMessage({ id: 'multicluster.memberClusters' })} <span className="num">({clusterList?.length})</span>
      </p>
      <div className="host_cluster_table">
        <Form className="searchFormCluster" form={clusterForm}>
          <Form.Item name="cluster_name" className="search_input">
            <Input.Search
              placeholder={intl.formatMessage({ id: 'multicluster.searchClusterName' })}
              onSearch={() => handleSearchClusterment()}
              autoComplete="off"
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button className="primary_btn">
                <Link
                  to={{
                    pathname: `/edit`,
                    state: {
                      type: intl.formatMessage({ id: 'multicluster.addCluster' }),
                      detailInfo: { labelList: [{ key: '', value: '' }] },
                    },
                  }}
                >
                  {intl.formatMessage({ id: 'multicluster.addCluster' })}
                </Link>
              </Button>
            </Space>
          </Form.Item>
        </Form>
        <Flex gap="middle" vertical>
          <Spin tip={intl.formatMessage({ id: 'multicluster.removingManagement' })} spinning={delLoading}>
            <Table
              style={{ height: 'calc(100vh - 378px)' }}
              columns={clusterColumns}
              dataSource={clusterList}
              locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
              pagination={{
                className: 'page',
                current: clusterPage,
                showTotal: (total) => intl.formatMessage({ id: 'multicluster.totalItems' }, { total }),
                showSizeChanger: true,
                showQuickJumper: true,
                pageSizeOptions: [10, 20, 50],
                onChange: (page) => setClusterPage(page),
              }}
              scroll={{ x: 1280 }}
            />
          </Spin>
        </Flex>
      </div>
      {isUpdate && (
        <Modal
          className="config_modal"
          width={1200}
          height={720}
          getContainer={false}
          title="kubeconfig"
          open={isShowCertificate}
          onCancel={handleCancel}
          footer={[
            <Button className="primary_btn" onClick={handleCancel}>
              {intl.formatMessage({ id: 'common.ok' })}
            </Button>,
          ]}
        >
          <CodeMirrorEditor yamlData={certificateInfo} isEdit={false} ref={childCodeMirrorRef} />
        </Modal>
      )}
      <AnnotationModal
        open={isLabelModalOpen}
        type="label"
        dataList={labels}
        callbackOk={handleLabelOkConfigurationDetail}
        callbackCancel={handleLabelCancelDetail}
      />
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'multicluster.removeManagement' })}
        open={clusterDelModalOpen}
        cancelFn={handleDelClusterCancel}
        content={[
          intl.formatMessage({ id: 'multicluster.removeWarning' }),
          intl.formatMessage({ id: 'multicluster.confirmRemoveCluster' }, { name: clusterDelName }),
        ]}
        isCheck={isClusterDelCheck}
        showCheck={true}
        checkFn={handleDeploymentCheckFn}
        confirmFn={handleClustertConfirm}
      />
    </div>
  );
}
