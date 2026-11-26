# pipeline 分支:Shogun 数据处理引擎

本分支只包含 Vicon Shogun Post 的自动化处理引擎(权威源码,4DHOI 项目同源):

```
scripts/extract_groups.py     FBX 7700 字节级解析/图构建(pipeline_split 依赖)
scripts/shogun/               处理引擎与 Web 胶水层
  pipeline_*.py               26 阶段流水线(解析/配置/重建/标记修复/解算/逐人导出)
  pipeline_session.py         ShogunPostCL 会话隔离(端口/所有权/SDK 加载)
  fbx_lightweight.py          净空裁剪(字节级 FBX 重写,M_Pruning)
  mesh_fbx.py                 FBX 解析/拼接/校验
  sequence_inspect.py         只读 MCP/FBX 名单与挂钟时码解析
  web_manager.py / web_worker.py  队列引擎(MocapStudio server 复用)
  theme_roster.py / step1_process_takes.py / test_*.py
```

运行方式见 main 分支 README;main(server+workbench)以 `--scripts <此目录>/shogun` 使用本引擎。
