/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, useState, useEffect, useStore } from 'openinula';
import { EditOutlined } from '@ant-design/icons';
import { Tag, message } from 'antd';
import { solveAnnotationOrLabelDiff } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import { ResponseCode } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import { editAnnotationsOrLabels } from '@/api/schedulingApi';
import { useIntl } from 'inula-intl';

export default function PriorityDetail({ details, refreshFn }) {
  const intl = useIntl();
  const [messageApi, contextHolder] = message.useMessage();
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [oldAnnotations, setOldAnnotataions] = useState(
    details?.metadata?.annotations ? details?.metadata?.annotations : '',
  );
  const [oldLabels, setOldLabels] = useState(details?.metadata?.labels ? details?.metadata?.labels : '');

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(intl.formatMessage({ id: 'annotation.noChanges' }));
      setIsAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      data.map((item) => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(intl.formatMessage({ id: 'annotation.duplicateKey' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels(details.metadata?.name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            messageApi.success(intl.formatMessage({ id: 'annotation.editSuccess' }));
            setTimeout(() => {
              refreshFn();
              setIsAnnotationModalOpen(false);
            }, 1000);
          }
        } catch (error) {
          if (error.response.status === 403) {
            message.error(intl.formatMessage({ id: 'common.permissionDenied' }));
          } else {
            message.error(`${intl.formatMessage({ id: 'annotation.editFailed' })}${error.response.data.message}`);
          }
        }
      }
    }
  };
  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(intl.formatMessage({ id: 'annotation.noLabelChanges' }));
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map((item) => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(intl.formatMessage({ id: 'annotation.duplicateKey' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels(details.metadata?.name, addLabelList);
          if (res.status === ResponseCode.OK) {
            messageApi.success(intl.formatMessage({ id: 'annotation.editLabelSuccess' }));
            setTimeout(() => {
              refreshFn();
              setIsLabelModalOpen(false);
            }, 1000);
          }
        } catch (error) {
          if (error.response.status === 403) {
            message.error(intl.formatMessage({ id: 'common.permissionDenied' }));
          } else {
            message.error(`${intl.formatMessage({ id: 'annotation.editLabelFailed' })}${error.response.data.message}`);
          }
        }
      }
    }
  };

  // 标签失败回调
  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  return (
    <Fragment>
      {contextHolder}
      <div className={`tab_container container_margin_box normal_container_height`} style={{ margin: '20px 32px' }}>
        <div className="detail_card">
          <h3>{intl.formatMessage({ id: 'priority.basicInfo' })}</h3>
          <div className="detail_info_box">
            <div className="base_info_list">
              <div className="flex_item_opt">
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'priority.priorityClassNameLabel' })}</p>
                  <p className="base_value">{details.metadata?.name}</p>
                </div>
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'priority.priorityValueLabel' })}</p>
                  <p className="base_value">{details.value}</p>
                </div>
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'priority.isDefaultLabel' })}</p>
                  <p className="base_value">
                    {details.globalDefault
                      ? intl.formatMessage({ id: 'common.yes' })
                      : intl.formatMessage({ id: 'common.no' })}
                  </p>
                </div>
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'priority.isPreemptible' })}</p>
                  <p className="base_value">
                    {details.preemptionPolicy === 'Never'
                      ? intl.formatMessage({ id: 'priority.notPreemptible' })
                      : intl.formatMessage({ id: 'priority.preemptible' })}
                  </p>
                </div>
              </div>
            </div>
            <div className="annotation">
              <div className="ann_title">
                <p>{intl.formatMessage({ id: 'priority.labels' })}</p>
                <EditOutlined
                  className="primary_icon common_antd_icon"
                  onClick={() => {
                    setIsLabelModalOpen(true);
                  }}
                />
              </div>
              <AnnotationModal
                open={isLabelModalOpen}
                type="label"
                dataList={details.metadata?.labels}
                callbackOk={handleLabelOk}
                callbackCancel={handleLabelCancel}
              />
              <div className="key_value">
                {details.metadata?.labels?.length ? (
                  details.metadata?.labels.map((item) => (
                    <LabelTag labelKey={item.key} labelValue={item.value} theme="light" />
                  ))
                ) : (
                  <span style={{ marginTop: '13px' }}>0个</span>
                )}
              </div>
            </div>
            <div className="annotation">
              <div className="ann_title">
                <p>{intl.formatMessage({ id: 'priority.annotations' })}</p>
                <EditOutlined
                  className="primary_icon common_antd_icon"
                  onClick={() => {
                    setIsAnnotationModalOpen(true);
                  }}
                />
              </div>
              <AnnotationModal
                open={isAnnotationModalOpen}
                type="annotation"
                dataList={details.metadata?.annotations}
                callbackOk={handleAnnotationOk}
                callbackCancel={handleAnnotationCancel}
              />
              <div className="key_value">
                {details.metadata?.annotations?.length ? (
                  details.metadata?.annotations.map((item) => (
                    <LabelTag labelKey={item.key} labelValue={item.value} theme="light" />
                  ))
                ) : (
                  <span style={{ marginTop: '13px' }}>{intl.formatMessage({ id: 'priority.zeroItems' })}</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Fragment>
  );
}
