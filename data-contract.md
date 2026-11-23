# 资料口径

对象是模拟会谈的教学支持。量表 0 至 100，低于 60 定为需额外支持；独立后测开放于初测后第六日，第八日到期，使用另一题本。历史上的 `prob` 是邀约概率，不能作为接受概率。P0 是旧策略，未为受规定限制的人群随机安排；P1 与 P2 在分别登记的支持范围内按教学组给定概率邀约，没有随机分配具体导师。相同教练关联的结果不当成独立观察。`short` 每次最多一席每人且仅用于解释分项的答疑，`full` 每人要完成间隔 1 至 3 日的两次练习且须同一位导师主持，两次必须在同一学员当前轮转截止前。已接受的答疑可与整段训练并行，但彼此不能代替。

数据表使用 UTF-8 CSV，第一行是字段名。同一业务号的引用指数据字典约定的原始对象；空白表示来源未提供，集合字段由逗号分隔后整体按 CSV 转义。
- `student.csv` 的字段为 `id`、`wave`、`track`、`rotation_end`、`policy`。
- `first_score.csv` 的字段为 `id`、`student`、`skill`、`value`、`received`、`status`、`supersedes`。
- `invitation.csv` 的字段为 `student`、`policy`、`assigned`、`offered_kind`、`prob`、`completed`、`coach`、`sent`。
- `post_score.csv` 的字段为 `student`、`book`、`observed`、`received`、`value`、`status`。
- `policy.csv` 的字段为 `id`、`tracks`、`kind`、`review_at_day`、`minimum_short_fraction`、`full_need_score`。
- `slot.csv` 的字段为 `id`、`coach`、`day`、`begin`、`capacity`、`kind`、`skills`。
- `appointment.csv` 的字段为 `id`、`student`、`slot`、`state`、`received`。
