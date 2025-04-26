
# 运行
```
pip install -r requierments.txt
python six_hat_bot.py -v
```

# 六顶思考帽多Agent分析系统架构说明
## 系统架构概述
本系统基于六顶思考帽理论，采用多Agent协作方式，对需求进行多角度、全方位分析。系统核心由六种思考帽Agent、信息搜集Agent、报告生成Agent、工具管理器、共享内存和模型API接口组成。

---

## 系统架构图（PlantUML）
```plantuml
@startuml
!theme spacelab
actor 用户 as User
User --> 系统 : 输入需求
package "六顶思考帽系统" {
  [六顶思考帽系统] as System
  [共享内存] as Memory
  [工具管理器] as ToolMgr
  [模型API] as ModelAPI
  [蓝帽Agent] as Blue
  [白帽Agent] as White
  [红帽Agent] as Red
  [黄帽Agent] as Yellow
  [黑帽Agent] as Black
  [绿帽Agent] as Green
  [信息Agent] as Info
  [报告Agent] as Report
  System --> Blue
  System --> White
  System --> Red
  System --> Yellow
  System --> Black
  System --> Green
  System --> Info
  System --> Report
  Blue ..> Memory
  White ..> Memory
  Red ..> Memory
  Yellow ..> Memory
  Black ..> Memory
  Green ..> Memory
  Info ..> ToolMgr
  ToolMgr ..> ModelAPI
  Report ..> Memory
}
系统 --> 用户 : 输出报告
@enduml
```

---

## 典型分析流程说明
1. 用户输入需求。
2. 系统保存需求到共享内存。
3. 蓝帽Agent设计分析流程。
4. 信息Agent自动检索相关信息。
5. 白/红/黄/黑帽Agent并行分析不同维度。
6. 绿帽Agent进行创新思维分析。
7. 报告Agent整合所有分析结果，生成最终报告。
8. 系统输出报告给用户。

---

## 分析流程图（PlantUML）
```plantuml
@startuml
!theme spacelab
start
:用户输入需求;
:系统保存需求到共享内存;
:蓝帽Agent设计分析流程;
:信息Agent检索信息;
fork
  :白帽Agent分析事实;
  :红帽Agent分析情感;
  :黄帽Agent分析价值;
  :黑帽Agent分析风险;
end fork
:绿帽Agent创新分析;
:报告Agent整合生成报告;
:系统输出报告;
stop
@enduml
```

---
如需详细类方法说明，请参考 six_hat_bot.py 源码注释。

## 优化后的架构图
以下是优化后的系统架构图，增加了反思Agent、评估模块，并明确了Agent间的交互：

```plantuml
@startuml
!theme spacelab
actor 用户 as User
User --> 系统 : 输入需求
package "六顶思考帽系统" {
  [六顶思考帽系统] as System
  [共享内存] as Memory
  [工具管理器] as ToolMgr
  [模型API] as ModelAPI
  [蓝帽Agent] as Blue
  [白帽Agent] as White
  [红帽Agent] as Red
  [黄帽Agent] as Yellow
  [黑帽Agent] as Black
  [绿帽Agent] as Green
  [信息Agent] as Info
  [反思Agent] as Reflect
  [评估模块] as Eval
  [报告Agent] as Report
  System --> Blue
  System --> White
  System --> Red
  System --> Yellow
  System --> Black
  System --> Green
  System --> Info
  System --> Reflect
  System --> Eval
  System --> Report
  Blue --> Memory
  White --> Memory
  Red --> Memory
  Yellow --> Memory
  Black --> Memory
  Green --> Memory
  Info --> ToolMgr
  ToolMgr --> ModelAPI
  Reflect --> Memory
  Eval --> Memory
  Report --> Memory
  Blue --> Reflect : 协调
  Reflect --> White : 反馈
  Reflect --> Red : 反馈
  Reflect --> Yellow : 反馈
  Reflect --> Black : 反馈
  Reflect --> Green : 反馈
  Eval --> Report : 评分
}
系统 --> 用户 : 输出报告
@enduml
```

## 优化后的分析流程
```plantuml
@startuml
!theme spacelab
start
:用户输入需求;
:系统保存需求到共享内存;
:蓝帽Agent设计分析流程;
while (迭代未结束?)
  :信息Agent检索信息;
  fork
    :白帽Agent分析事实;
    :红帽Agent分析情感;
    :黄帽Agent分析价值;
    :黑帽Agent分析风险;
    :绿帽Agent创新分析;
  end fork
  :反思Agent审视结果;
  :蓝帽Agent调整流程;
end while
:评估模块评分结果;
:报告Agent整合生成报告;
:系统输出报告;
stop
@enduml
```