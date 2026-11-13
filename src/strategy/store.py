"""데이터로 들어가는 문 (11장). decide는 이 Store를 통해서만 데이터를 본다.
reports(before)가 as_of 전날까지 게시된 리포트만 돌려주므로, 그 뒤 코드가 어떤 실수를 해도 미래 리포트는 보이지 않는다."""
import numpy as np
import pandas as pd


class Store:
    def __init__(self, reports, daily, universe):
        """reports: nid, write_date, broker_code, item_code (+ grade). daily: date, item_code, ret. universe: date, item_code."""
        r = reports.sort_values(["write_date", "nid"]).reset_index(drop=True)
        self._reports, self._dates = r, r["write_date"].to_numpy()
        self.days = pd.DatetimeIndex(np.sort(daily["date"].unique()))      # 거래일 달력
        self._ret = daily.pivot(index="date", columns="item_code", values="ret").reindex(self.days).fillna(0.0)
        self._lists = {d: g.to_numpy() for d, g in universe.groupby("date")["item_code"]}
        self._list_days = pd.DatetimeIndex(sorted(self._lists))

    def with_grades(self, grades):
        """r_nid → grade 표를 붙인 새 Store. 등급은 인자로 받는다 (18장: 무작위 라벨도 같은 길로 들어온다)."""
        new = object.__new__(Store)
        new.__dict__.update(self.__dict__)
        new._reports = self._reports.assign(grade=self._reports["nid"].map(grades.set_index("r_nid")["grade"]))
        return new

    def reports(self, before):
        """게시일이 before보다 앞인 리포트만 (게시일·nid 순)."""
        return self._reports.iloc[:np.searchsorted(self._dates, np.datetime64(pd.Timestamp(before)), side="left")]

    def prices(self, until):
        """until까지의 일별 등락률 표 (날짜 × 종목). 데이터가 없는 날은 0 (16장)."""
        return self._ret.loc[:until]

    def universe(self, on):
        """on 날짜의 명단. 그날 명단이 없으면 직전 거래일 명단 (16장)."""
        i = self._list_days.searchsorted(pd.Timestamp(on), side="right") - 1
        return self._lists[self._list_days[i]] if i >= 0 else np.array([], dtype=object)
