/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/log.less';
import { useIntl } from 'inula-intl';
import { Input, Button, Table, Space } from 'antd';
import { useState, useEffect } from 'openinula';
import ModalComponent from '@/components/ModalComponent';
import { CollectionCreateForm } from '@/components/CollectionCreateForm';
import { ResponseCode } from '@/common/constants';
import { getLogAlartData } from '@/api/containerLogApi';
import { solveAnnotation, sorterFirstAlphabet } from '@/tools/utils';
const { Search } = Input;

export default function LogAlerts() {
  const intl = useIntl();
  const [alartDataSource, setAlartDataSource] = useState([]);
  const [alarttableSource, setAlarttableSource] = useState([]);
  const onSearchLogAlert = (value) => {
    if (value) {
      setAlarttableSource(alartDataSource.filter((item) => item.name.includes(value)));
    } else {
      setAlarttableSource(alartDataSource);
    }
  };
  const columns = [
    {
      title: intl.formatMessage({ id: 'alert.name' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
    },
    {
      title: intl.formatMessage({ id: 'alert.expression' }),
      dataIndex: 'expr',
      key: 'expr',
    },
    {
      title: intl.formatMessage({ id: 'alert.duration' }),
      dataIndex: 'time',
      key: 'time',
      sorter: (a, b) => sorterFirstAlphabet(a.time, b.time),
    },
    {
      title: intl.formatMessage({ id: 'alert.labels' }),
      dataIndex: 'labels',
      key: 'labels',
      render: (_, record) => (
        <Space style={{ display: 'flex', flexWrap: 'wrap' }}>
          {record.labels.map((item) => (
            <p
              style={{ fontSize: '14px', padding: '5px 12px', background: '#cfe7ffff', color: '#333', borderRadius: 6 }}
            >
              {item}
            </p>
          ))}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'alert.annotations' }),
      dataIndex: 'annotations',
      key: 'annotations',
      width: '30%',
      render: (_, record) => (
        <Space style={{ display: 'flex', flexWrap: 'wrap' }}>
          {record.annotations.map((item) => (
            <p
              style={{ fontSize: '14px', padding: '5px 12px', background: '#cfe7ffff', color: '#333', borderRadius: 6 }}
            >
              {item}
            </p>
          ))}
        </Space>
      ),
    },
  ];

  const [openEdit, setOpenEdit] = useState(false);
  const [openAdd, setOpenAdd] = useState(false);
  function handleEdit() {
    setOpenEdit((isOpen) => !isOpen);
  }
  function handleAddCancel() {
    setOpenAdd((isOpen) => !isOpen);
  }
  const handleCancelEdit = () => {
    setOpenEdit(false);
  };
  const onCreate = (values) => {
    setFormValues(values);
    setOpenEdit(false);
  };

  function handleAdd() {
    setOpenAdd((isOpen) => !isOpen);
  }
  const getLogAlartList = async () => {
    const res = await getLogAlartData();
    if (res.status === ResponseCode.OK) {
      let _data = res.data.data.RuleGroups;
      let _arr = [];
      _data.map((item) => {
        let labelList = solveAnnotation(item.Rules[0].Labels);
        let annotationsList = solveAnnotation(item.Rules[0].Annotations);
        let _labelList = [];
        let _annotationsList = [];
        labelList.map((labelItem) => {
          _labelList.push(`${labelItem.key}=${labelItem.value}`);
        });
        annotationsList.map((annotationsItem) => {
          _annotationsList.push(`${annotationsItem.key}:${annotationsItem.value}`);
        });
        _arr.push({
          name: item.Name,
          expr: item.Rules[0].Expr,
          time: item.Rules[0].For,
          labels: _labelList,
          annotations: _annotationsList,
        });
      });
      setAlartDataSource(_arr);
      setAlarttableSource(_arr);
    }
  };

  useEffect(() => {
    getLogAlartList();
  }, []);
  return (
    <div className="logAlerts">
      <div className="logAlerts_top">
        <Search
          placeholder={intl.formatMessage({ id: 'alert.searchPlaceholder' })}
          onSearch={onSearchLogAlert}
          style={{ width: 200 }}
        />
        <Button className="primary_btn_log" onClick={handleAdd} style={{ display: 'none' }}>
          {intl.formatMessage({ id: 'alert.addRule' })}
        </Button>
      </div>
      <div>
        <Table
          dataSource={alarttableSource}
          columns={columns}
          pagination={false}
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
        />
        <ModalComponent
          open={openEdit}
          title={intl.formatMessage({ id: 'alert.editRuleTitle' })}
          onCreate={onCreate}
          onCancel={handleCancelEdit}
          initialValues={{ modifier: 'public' }}
        >
          <CollectionCreateForm
            onFormInstanceReady={(instance) => {
              setFormInstance(instance);
            }}
          />
        </ModalComponent>
        <ModalComponent
          open={openAdd}
          title={intl.formatMessage({ id: 'alert.addRuleTitle' })}
          onCreate={onCreate}
          onCancel={handleAddCancel}
          initialValues={{ modifier: 'public' }}
        >
          <CollectionCreateForm
            onFormInstanceReady={(instance) => {
              setFormInstance(instance);
            }}
          />
        </ModalComponent>
      </div>
    </div>
  );
}
