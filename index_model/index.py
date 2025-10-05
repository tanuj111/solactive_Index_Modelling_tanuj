import datetime as dt
from pathlib import Path
import pandas as pd

class IndexModel:
    def __init__(self) -> None:
        # To be implemented
        root = Path(__file__).resolve().parents[1]
        print(root)
        prices_csv_file_location = root/"data_sources/stock_prices.csv"
        self.underlying_data=pd.read_csv(prices_csv_file_location, index_col="Date", parse_dates=['Date'], dayfirst=True).sort_index()
        self.underlyings=self.underlying_data.columns
        self.rebalDates=None
        self.rebalReturns=None
        self.weights=None
        self.indexLvls = None        
        self.portfolioPerformance = None

    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        # To be implemented
        self._rebal_dates(start_date, end_date)
        self._weights_calc()
        # computing returns for each security since last rebal
        rebal_returns = self._compute_rebal_returns(start_date)
        self._calculate_index_level(start_date)        

    def _calculate_index_level(self, startDt):
        # to compute the index level we first computed the rebal level on each of the rebalance dates: rebalLvls
        # then to compute the index levels we multiply the total weighted return on a given day with the level on the previous rebalance date
        rebalReturns = pd.Series(self.rebalReturns[pd.to_datetime(self.rebalDates)])
        rebalLvls_ = pd.DataFrame(rebalReturns.cumprod()*100.0)
        rebalLvls = rebalLvls_.reindex(self.underlying_data.index,method="ffill")
        rebalLvlsShifted = rebalLvls.shift(1)
        rebalLvlsShifted = rebalLvlsShifted[rebalLvlsShifted.index >= pd.to_datetime(startDt)]
        rebalLvlsShifted.loc[pd.to_datetime(startDt)]=100 # setting initial index level to 100
        indexLvls=(rebalLvlsShifted*pd.DataFrame(self.rebalReturns))
        indexLvls.columns = ["IndexLvls"]
        self.indexLvls=indexLvls

    def _compute_rebal_returns(self, startDt):
        # rebal returns are the sum of weighted returns for individual securities.
        # we compute the returns for individual securities as (price(t)/price(previous rebal dt)-1)*weight_of_security
        rebalPrices = self.underlying_data.loc[pd.to_datetime(self.rebalDates)] # filtering for rebal prices
        rebalPricesReindexed = rebalPrices.reindex(self.underlying_data.index, method="ffill")
        rebalPricesShift = rebalPricesReindexed.shift(1)
        rebalReturns = self.underlying_data/rebalPricesShift-1
        rebalWeightedReturns = rebalReturns*self.weights
        rebalTotalReturns = rebalWeightedReturns.sum(axis=1)+1
        rebalTotalReturns = rebalTotalReturns[rebalTotalReturns.index >= pd.to_datetime(startDt)]
        self.rebalReturns = rebalTotalReturns
        
    def _rebal_dates(self, start_dt: dt.date, end_dt: dt.date) -> pd.Series:
        rebal_dates_ = self.underlying_data.resample("BMS").first().index
        # filtering the rebal dates to be after backtest_start and before backtest_end
        rebal_dates_ = [i.date() for i in rebal_dates_ if i.date() >= start_dt and i.date() <= end_dt]        
        rebal_dates = pd.Series(rebal_dates_)
        self.rebalDates=rebal_dates        

    def _weights_calc(self) -> pd.DataFrame:
        weightsDict = {}
        for dt in self.rebalDates:
            # Selection prices are stock prices as of rebal date - 1 biz day
            selectionPrices = self.underlying_data[self.underlying_data.index < pd.Timestamp(dt)].iloc[-1]
            # compute rank of the prices of all securities for the given rebal date
            # As all companies have same number of outstanding shares, the market cap ranking will be same as  price ranking            
            securitiesRank = selectionPrices.rank(ascending=False)
            weights_ = pd.Series(0, index=securitiesRank.index, dtype=float) # series with zero weights for all securities
            weights_.loc[securitiesRank==1] = 0.50 # assigning 50% to security with maximum price
            weights_.loc[(securitiesRank==2)|(securitiesRank==3)] = 0.25 # assigning 25% to security with rank 2 & 3
            weightsDict[dt] = weights_ # appending weights_ to our dictionary        
        
        # creating weights dataframe from weights dictionary
        weights_df = pd.DataFrame.from_dict(weightsDict, orient='index')
        weights_df = weights_df.reindex(self.underlying_data.index, method="ffill")
        weights_df = weights_df.shift(1).fillna(0) # moving forward the weights by 1 biz day as the weights will come in effect from rebal+1 day onwards.
        weights_df = weights_df[weights_df.index >= pd.to_datetime(self.rebalDates[0])]
        self.weights=weights_df
           
    def _processing_prices(self) -> pd.DataFrame:
        df = self.underlying_data
        # removing data for those dates which are not index business day
        df["Weekday"] = df.index.weekday
        df = df[df.Weekday < 5]
        df.drop(columns=["Weekday"], axis=1, inplace=True)
        return df

    def _portfolio_performance(self) -> pd.DataFrame:
        portfolioPerf={}
        print(f"portfolio_return is {type(self.indexLvls.index[-1]-self.indexLvls.index[0])}")
        diff_bw_first_and_last_index_dt = (self.indexLvls.index[-1]-self.indexLvls.index[0]).days
        portfolioPerf["Ann.Return"] = ((self.indexLvls.iloc[-1]/self.indexLvls.iloc[0])**(1/(365/diff_bw_first_and_last_index_dt))-1)*100
        portfolioPerf["Std.Dev"] = self.indexLvls.pct_change().dropna().std()*(252**0.5)*100
        portfolioPerf_df= pd.DataFrame.from_dict(portfolioPerf, orient='index')        
        portfolioPerf_df.columns=["Value"]
        self.portfolioPerformance = portfolioPerf_df["Value"].apply(lambda x: f"{x:.2f}%")
        

    def export_values(self, file_name: str) -> None:
        # generation csv file
        self.indexLvls.to_csv(file_name,index=True, )

        # including portfolio performance
        self._portfolio_performance()

        # the code also generated an xlsx file which will have the securities' prices, rebal dates, rebal returns, weights, index levels
        startDt=self.indexLvls.head(1).index
        endDt=self.indexLvls.tail(1).index
        excel_file = f"Index_Calculations_start-{pd.to_datetime(startDt)[0].strftime('%Y%m%d')}_end-{pd.to_datetime(endDt)[0].strftime('%Y%m%d')}.xlsx"
        print(excel_file)
        xlwrite = pd.ExcelWriter(excel_file, engine='xlsxwriter')
        self.underlying_data.to_excel(xlwrite, sheet_name = "prices", index=True)
        self.rebalDates.rename("RebalDates").to_excel(xlwrite, sheet_name = "Rebal_Dates", index=False)
        self.weights.to_excel(xlwrite, sheet_name = "Weights", index=True,)
        self.rebalReturns.rename("RebalReturns").to_excel(xlwrite, sheet_name = "Rebal_rets", index=True)
        self.indexLvls.to_excel(xlwrite, sheet_name = "Index_Lvls", index=True)
        self.portfolioPerformance.to_excel(xlwrite, sheet_name = "Index_Performance", index=True)
        xlwrite.close()

        
