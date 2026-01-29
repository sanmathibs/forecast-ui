import json
from typing import Dict, Set
import matplotlib

# Use non-GUI backend to avoid Tk warnings when run inside API workers/tests.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import date, datetime, timedelta
from dateutil import easter
from sklearn.metrics import mean_squared_error


def get_week_of_year(dt_col: pd.Series):
    col_woy = dt_col.dt.strftime("%U").astype(int)
    for y in dt_col.dt.year.unique().tolist():
        # get WoY of 1 Jan of the year
        d = date(y, 1, 1)
        woy = d.strftime("%U")
        if int(woy) == 0:
            # increase WoY of this year by 1
            col_woy[dt_col.dt.year == y] += 1
    return col_woy


class Predict:

    def assign_event_name(self, dt: datetime):
        """Return an event name like 'Christmas +2' or 'Black Friday -1' or None."""
        y = dt.year
        d = dt.date()

        # Father's Day: first Sunday of September
        sept1 = date(y, 9, 1)
        first_sunday_sept = sept1 + timedelta(days=(6 - sept1.weekday()) % 7)
        diff_fd = (d - first_sunday_sept).days
        if -1 <= diff_fd <= -1:
            if diff_fd == -0:
                return "Father's Day 0"
            return f"Father's Day {diff_fd:+d}"

        # Black Friday the last Friday of November
        nov1 = date(y, 11, 1)
        fridays = [
            nov1 + timedelta(days=i)
            for i in range(30)
            if (nov1 + timedelta(days=i)).weekday() == 4
        ]
        if len(fridays) >= 4:
            bf = fridays[-1]  # The last Friday
            diff_bf = (d - bf).days
            if -1 <= diff_bf <= 3:
                if diff_bf == 0:
                    return "Black Friday 0"
                return f"Black Friday {diff_bf:+d}"

        # Christmas cluster
        christmas = date(y, 12, 25)
        diff_x = (d - christmas).days
        if -14 <= diff_x <= 3:
            if diff_x == 0:
                return "Christmas 0"
            # diff_x = int((diff_x - int(diff_x / abs(diff_x)))/3) + int(diff_x / abs(diff_x))
            return f"Christmas {diff_x:+d}"

        # New Year
        if d == date(y, 1, 1):
            return "New Year 0"

        # Easter cluster (use dateutil.easter)
        easter_sun = easter.easter(y)
        diff_e = (d - easter_sun).days
        if -2 <= diff_e <= 1:
            # Good Friday is easter -2 -> map to "Good Friday 0" if desired
            if diff_e == -2:
                return "Good Friday 0"
            if diff_e == 0:
                return "Easter 0"
            return f"Easter {diff_e:+d}"

        # ANZAC Day and nearby Friday
        anzac = date(y, 4, 25)
        diff_a = (d - anzac).days
        if -3 <= diff_a <= 3:
            if diff_a == 0:
                return "ANZAC Day" + (" Shutdown" if y >= 2025 else "")
            # elif d.weekday() == 4:
            #     return "Friday Near ANZAC Day"

        return None

    def day_name(self, dt: datetime):
        return dt.strftime("%A")

    def get_week_of_year(self, dt_col: pd.Series):
        col_woy = dt_col.dt.strftime("%U").astype(int)
        for y in dt_col.dt.year.unique().tolist():
            # get WoY of 1 Jan of the year
            d = date(y, 1, 1)
            woy = d.strftime("%U")
            if int(woy) == 0:
                # increase WoY of this year by 1
                col_woy[dt_col.dt.year == y] += 1
        return col_woy

    def predict(
        self, df_sales: pd.DataFrame, start_forecast: datetime, end_forecast: datetime
    ):
        """_summary_

        This is a Sales forecasting function using Random Forest.
        Future sales are predicted based on historical data.
        Features that might affect sales include:
            - Holidays or special events
            - Year
            - Week of the year
            - The day of the week

        Args:
            df_sales (pd.DataFrame): columns=[Date,Sales]
            start (datetime): start date of prediction
            end (datetime): end date of prediction

        Returns:
            _type_: future_prediction, historical_prediction
        """
        print("Starting Sales Prediction using Random Forest...")

        # Utility: signed days to nearest Christmas (Dec 25). Before is negative, after is positive.
        def days_to_nearest_christmas(dt: pd.Timestamp) -> int:
            year = int(dt.year)
            c_prev = pd.Timestamp(year=year - 1, month=12, day=25)
            c_this = pd.Timestamp(year=year, month=12, day=25)
            c_next = pd.Timestamp(year=year + 1, month=12, day=25)
            deltas = [
                (dt - c_prev).days,
                (dt - c_this).days,
                (dt - c_next).days,
            ]
            # choose the delta with the smallest absolute value; sign conveys before/after
            return min(deltas, key=lambda d: abs(d))

        # -----------------------
        # Prepare training data
        # -----------------------
        df_sales["DayName"] = df_sales["Date"].dt.day_name()
        df_sales["Year"] = df_sales["Date"].dt.year
        df_sales["Month"] = df_sales["Date"].dt.month
        df_sales["WoY"] = self.get_week_of_year(df_sales["Date"])
        df_sales["SpecialEvent"] = df_sales["Date"].apply(
            lambda dt: self.assign_event_name(dt)
        )
        df_sales["Days_To_Nearest_Christmas"] = df_sales["Date"].apply(
            days_to_nearest_christmas
        )
        shutdown_events = [
            str(e) for e in df_sales.loc[df_sales["Sales"] < 2000, "SpecialEvent"].unique() if pd.notna(e)
        ]
        print(f"Shutdown events: {(', '.join(shutdown_events))}")
        df_sales["Is_Shutdown"] = (
            df_sales["SpecialEvent"].isin(shutdown_events).astype(int)
        )
        df_sales["Is_Weekend"] = (
            df_sales["DayName"].isin(["Saturday", "Sunday"]).astype(int)
        )
        df_sales.loc[df_sales["Is_Shutdown"] > 0, "SpecialEvent"] = (
            np.nan
        )  # Replace event name with NaN for shutdown days
        # Add a new feature for special event and weekend interaction
        df_sales["Is_SpecialEvent_Weekend"] = (
            (df_sales["SpecialEvent"].notna()) & (df_sales["Is_Weekend"] == 1)
        ).astype(int)
        # df_sales.loc[df_sales["Is_SpecialEvent_Weekend"] ==1, "Is_Weekend"] = 0 # Replace weekend with 0 for shutdown days
        # df_sales.loc[df_sales["Is_SpecialEvent_Weekend"] ==1, "SpecialEvent"] = np.nan # Replace event name with NaN for shutdown days
        # df_sales.loc[:, "SpecialEvent"] = df_sales["SpecialEvent"].apply(lambda e: ((e.split('-')[0]+'Early') if '-' in e else (e.split('+')[0]+'Late') if '+' in e else e) if isinstance(e, str) else e)

        # Encode categorical DayName
        dow_dummies_sales = pd.get_dummies(df_sales["DayName"], prefix="DoW")
        # Encode categorical Event
        event_dummies_sales = pd.get_dummies(df_sales["SpecialEvent"], prefix="SE")

        # Feature set
        X = pd.concat(
            [
                df_sales[
                    [
                        "Year",
                        "WoY",
                        "Month",
                        "Days_To_Nearest_Christmas",
                        "Is_Shutdown",
                        # "Is_Weekend",
                        "Is_SpecialEvent_Weekend",
                    ]
                ],
                dow_dummies_sales,
                event_dummies_sales,
            ],
            axis=1,
        )

        y = df_sales["Sales"]

        # Use TimeSeriesSplit for time series cross-validation
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

        n_samples = len(X)
        horizon_days = int(max(7, (end_forecast - start_forecast).days + 1))  # e.g., 30
        print(f"Forecast horizon days: {horizon_days}")

        # CV sizing: match original behaviour when feasible, but fall back if data is short.
        cv_test_size = horizon_days
        gap = 0
        min_train_required = 6 * 30  # ~6 months history
        feasible = (n_samples - min_train_required) // cv_test_size if n_samples > min_train_required else 0

        if feasible < 2:  # not enough data for 3+ splits at full horizon size
            cv_test_size = max(7, min(horizon_days, n_samples // 4 or 7))
            feasible = max(2, (n_samples - cv_test_size * 3) // cv_test_size)

        # additional guard: can't have more splits than possible given test_size
        max_splits_possible = max(2, n_samples // (cv_test_size * 2))
        n_splits = max(2, min(5, max_splits_possible, int(feasible) if feasible else 2))

        print(
            f"Using TimeSeriesSplit with n_splits={n_splits} test_size={cv_test_size} and gap={gap}."
        )
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=cv_test_size, gap=gap)

        param_grid = {
            "n_estimators": [300, 400],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2],
        }
        rf_base = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
        )
        print("Starting GridSearchCV for Random Forest...")
        grid_search.fit(X, y)
        print(
            f"Best parameters from GridSearchCV (TimeSeriesSplit): {grid_search.best_params_}"
        )
        rf = grid_search.best_estimator_
        print("Fitting Random Forest on full historical data...")
        historial_pred = rf.predict(X)

        # Get feature importances
        feature_importances = rf.feature_importances_

        # Get feature names
        feature_names = X.columns

        # Combine feature names and importances
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)

        print(feature_importance_df)

        # replace sales forecast for shutdown events
        historial_pred[df_sales[df_sales["Is_Shutdown"] == 1].index] = 0
        print(
            "Historical forecast RMSE:",
            mean_squared_error(df_sales["Sales"], historial_pred),
        )
        df_sales_ = df_sales[
            ["Date", "Sales", "DayName", "Year", "WoY", "SpecialEvent"]
        ].copy()
        df_sales_["SalesForecast"] = historial_pred
        df_sales_.rename(columns={"Sales": "SalesActual"}, inplace=True)
        # df_sales_.to_csv("df_sales_with_forecast.csv", index=False)

        # Use the last split as validation for reporting
        val_indices = list(tscv.split(X))[-1][1]
        X_val = X.iloc[val_indices]
        y_val = y.iloc[val_indices]
        y_val_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
        mae = np.mean(np.abs(y_val - y_val_pred))
        avg_actual = np.mean(y_val)
        rel_rmse = rmse / avg_actual * 100

        print(f"Random Forest validation RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Relative RMSE: {rel_rmse:.3f}% of average actual value")

        # -----------------------
        # Predict for forecast period
        # -----------------------
        df_forecast = pd.DataFrame(
            {"Date": pd.date_range(start_forecast, end_forecast, freq="D")}
        )
        df_forecast["DayName"] = df_forecast["Date"].dt.day_name()
        df_forecast["Year"] = df_forecast["Date"].dt.year
        df_forecast["Month"] = df_forecast["Date"].dt.month
        df_forecast["WoY"] = self.get_week_of_year(df_forecast["Date"])
        df_forecast["SpecialEvent"] = df_forecast["Date"].apply(
            lambda dt: self.assign_event_name(dt)
        )
        df_forecast["Days_To_Nearest_Christmas"] = df_forecast["Date"].apply(
            days_to_nearest_christmas
        )
        df_forecast["Is_Shutdown"] = (
            df_forecast["SpecialEvent"].isin(shutdown_events).astype(int)
        )
        df_forecast["Is_Weekend"] = (
            df_forecast["DayName"].isin(["Saturday", "Sunday"]).astype(int)
        )
        df_forecast.loc[df_forecast["Is_Shutdown"] > 0, "SpecialEvent"] = (
            np.nan
        )  # Replace event name with NaN for shutdown days
        df_forecast["Is_SpecialEvent_Weekend"] = (
            (df_forecast["SpecialEvent"].notna()) & (df_forecast["Is_Weekend"] == 1)
        ).astype(int)
        # df_forecast.loc[df_forecast["Is_SpecialEvent_Weekend"] == 1, "Is_Weekend"] = 0
        # df_forecast.loc[df_forecast["Is_SpecialEvent_Weekend"] ==1, "SpecialEvent"] = np.nan
        # df_forecast.loc[:, "SpecialEvent"] = df_forecast["SpecialEvent"].apply(lambda e: ((e.split('-')[0]+'Early') if '-' in e else (e.split('+')[0]+'Late') if '+' in e else e) if isinstance(e, str) else e)

        dow_dummies_forecast = pd.get_dummies(df_forecast["DayName"], prefix="DoW")
        event_dummies_forecast = pd.get_dummies(
            df_forecast["SpecialEvent"], prefix="SE"
        )
        # Ensure same columns
        for col in X.columns:
            if col not in dow_dummies_forecast.columns and col.startswith("DoW_"):
                dow_dummies_forecast[col] = 0
            if col not in event_dummies_forecast.columns and col.startswith("SE_"):
                event_dummies_forecast[col] = 0

        X_future = pd.concat(
            [
                df_forecast[
                    [
                        "Year",
                        "WoY",
                        "Month",
                        "Days_To_Nearest_Christmas",
                        "Is_Shutdown",
                        # "Is_Weekend",
                        "Is_SpecialEvent_Weekend",
                    ]
                ],
                dow_dummies_forecast,
                event_dummies_forecast,
            ],
            axis=1,
        )

        # Make sure future df and train df has the same columns order
        X_future = X_future[X.columns]

        # X.to_csv('X.csv')
        # X_future.to_csv('X_future.csv')
        # df_sales.to_csv('df_sales.csv', index=False)
        future_pred = rf.predict(X_future)
        future_pred[df_forecast[df_forecast["Is_Shutdown"] == 1].index] = 0
        # df_forecast["SalesForecast"] = future_pred
        # df_sales["SalesForecast"] = historial_pred
        # df_final = pd.concat([df_sales, df_forecast], axis=0)
        # df_final.to_csv("df_final_SalesForecast.csv", index=False)
        return future_pred, historial_pred


class QuantumForecast:
    """
    Encapsulates the quantum forecasting pipeline previously implemented as a script.

    Inputs/Outputs:
    - Reads sales and labour history from input_json (expects a 'Sale' array with fields: Date, Sales Actual, Sales Forecast, Hours Actual, Hours Forecast)
    - Writes two CSVs:
        - 'quantum_forecast.csv' for forecast period details
        - 'full_quantum_forecast_with_history.csv' with history + forecasts

    Key attributes after run():
    - df_sales: enriched historical sales DataFrame
    - df_forecast: forecast period DataFrame with hybrid forecasts
    - reg_results, smooth_slope, sales_bin
    """

    def __init__(
        self,
        input_json: str | None = None,
        intput_df: pd.DataFrame | None = None,
        quantum_hours: float = 6.0,
        foundation_state: Dict[str, int] | None = None,
        forecast_start: pd.Timestamp | str = None,
        forecast_end: pd.Timestamp | str = None,
        smooth_slope_digits: int = 5,
        blend_weight: float = 0.3,
        verbose: bool = True,
        name: str = "Default",
    ) -> None:
        self.input_json = input_json
        self._raw_input_df = intput_df
        self.quantum_hours = quantum_hours
        self.foundation_state = foundation_state or {
            "Monday": 3,
            "Tuesday": 3,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 3,
            "Saturday": 6,
            "Sunday": 5,
        }
        self.name = name
        self._safe_name = self.name.replace(" ", "_").replace("/", "_")
        try:
            self.forecast_start = pd.to_datetime(forecast_start)
        except:
            pass
        try:
            self.forecast_end = pd.to_datetime(forecast_end)
        except:
            pass
        # intput_df.to_csv(
        #     f"debug_input_df_{forecast_start.date()}_{forecast_end.date()}.csv",
        #     index=False,
        # )
        self.smooth_slope_digits = smooth_slope_digits
        self.blend_weight = blend_weight
        self.verbose = verbose

        # Will be populated during run()
        self.df_sales: pd.DataFrame | None = None
        self.df_forecast: pd.DataFrame | None = None
        self.reg_results: Dict[str, Dict[str, float]] | None = None
        self.smooth_slope: float | None = None
        self.sales_bin: float | None = None
        self.event_class: Dict[str, str] | None = None
        self.hist_ev: pd.DataFrame | None = None

    # -----------------------
    # Internal helpers
    # -----------------------
    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _load_data(self) -> pd.DataFrame:
        """Load sales and labour data from input JSON or provided DataFrame."""
        if self._raw_input_df is not None:
            self._log("Using provided input DataFrame.")
            df_sales = self._raw_input_df.copy()
            df_sales.reset_index(drop=True, inplace=True)
        else:
            with open(self.input_json, "r") as f:
                data = json.loads(f.read())
            df_sales = pd.json_normalize(data["Sale"])

        df_sales = df_sales[
            ["Date", "SalesActual", "SalesForecast", "HoursActual", "HoursForecast"]
        ].copy()

        df_sales["Date"] = pd.to_datetime(df_sales["Date"])
        df_sales["Year"] = df_sales["Date"].dt.year
        df_sales["Month"] = df_sales["Date"].dt.month
        df_sales["WoY"] = get_week_of_year(df_sales["Date"])
        df_sales["DayName"] = df_sales["Date"].dt.day_name()

        # Errors/stats
        df_sales["SalesForecast_Error"] = (
            df_sales["SalesActual"] - df_sales["SalesForecast"]
        )
        df_sales["Labour_Forecast_Error"] = (
            df_sales["HoursActual"] - df_sales["HoursForecast"]
        )

        # Determine default forecast window if not provided
        # If forecast_start or forecast_end were passed as None, pd.to_datetime(None) -> NaT
        last_sale_date: pd.Timestamp | None = None
        if not df_sales.empty and "Date" in df_sales.columns:
            last_sale_date = pd.to_datetime(df_sales["Date"]).max()

        # Start: next day after the last sale date when not provided
        if pd.isna(self.forecast_start):
            if last_sale_date is not None and not pd.isna(last_sale_date):
                self.forecast_start = last_sale_date + pd.Timedelta(days=1)
            else:
                # Fallback: if no sales data, start from today + 1 day
                self.forecast_start = pd.Timestamp.today().normalize() + pd.Timedelta(
                    days=1
                )

        # End: 30 days after the start when not provided
        if pd.isna(self.forecast_end):
            self.forecast_end = self.forecast_start + pd.Timedelta(days=30)

        # If only end was provided but it's before start, adjust to 30 days after start
        if self.forecast_end < self.forecast_start:
            self.forecast_end = self.forecast_start + pd.Timedelta(days=30)

        return df_sales

    def _fit_implied_labour(
        self, df_sales: pd.DataFrame
    ) -> tuple[Dict[str, Dict], float]:
        reg_results: Dict[str, Dict] = {}
        for dow in df_sales["DayName"].unique():
            sub = df_sales[df_sales["DayName"] == dow].dropna(
                subset=["SalesActual", "HoursActual"]
            )
            if len(sub) < 5:
                reg_results[dow] = {
                    "slope": np.nan,
                    "intercept": np.nan,
                    "std_err": np.nan,
                }
                continue
            X = sub[["SalesActual"]].values.reshape(-1, 1)
            y = sub["HoursActual"].values
            lm = LinearRegression().fit(X, y)
            preds = lm.predict(X)
            residuals: np.ndarray = y - preds
            std_err = residuals.std(ddof=1)
            reg_results[dow] = {
                "slope": float(lm.coef_[0]),
                "intercept": float(lm.intercept_),
                "std_err": float(std_err),
            }

        avg_slope = np.nanmean([v["slope"] for v in reg_results.values()])
        smooth_slope = round(avg_slope, self.smooth_slope_digits)

        # Force common slope, recompute per-DoW intercepts
        for dow in reg_results:
            sub = df_sales[df_sales["DayName"] == dow].dropna(
                subset=["SalesActual", "HoursActual"]
            )
            if len(sub) < 2:
                reg_results[dow].update(
                    {"slope_smoothed": smooth_slope, "intercept_smoothed": np.nan}
                )
                continue
            intercept_smoothed = (
                sub["HoursActual"] - smooth_slope * sub["SalesActual"]
            ).mean()
            reg_results[dow].update(
                {
                    "slope_smoothed": smooth_slope,
                    # round intercept to nearest even number
                    "intercept_smoothed": round(round(intercept_smoothed / 2.0) * 2.0),
                }
            )

        self._log("\nImplied labour regression (per DoW):")
        for dow in reg_results:
            self._log(f"  {dow}: {reg_results[dow]}")
        return reg_results, smooth_slope

    def _mark_special_and_events(
        self, df_sales: pd.DataFrame, sales_bin: float
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        # State from sales bin
        df_sales = df_sales.copy()
        df_sales.loc[df_sales["SalesActual"] <= 0, "State"] = 0

        def is_special(row):
            dow = row["DayName"]
            fs = self.foundation_state[dow]
            st = row["State"]
            return (st == 0) or (st >= fs + 4)

        df_sales["Is_Special_Event"] = df_sales.apply(is_special, axis=1)

        # Assign event names
        df_sales["EventName"] = df_sales["Date"].apply(self.assign_event_name)
        df_sales["EventName"] = df_sales["EventName"].fillna("Unclassified")

        # Classify events by weekday consistency
        event_weekdays = df_sales.groupby("EventName")["DayName"].nunique().to_dict()

        def classify_event(event_name: str) -> str:
            nwd = event_weekdays.get(event_name, 0)
            return "DoW_Plus" if nwd > 1 else "DoW_Independent"

        event_class = {ev: classify_event(ev) for ev in event_weekdays.keys()}
        return df_sales, event_class

    def _three_factor(self, df_sales: pd.DataFrame) -> tuple[Dict, Dict, Dict]:
        df_normal = df_sales[df_sales["EventName"] == "Unclassified"].copy()
        annual_factor = df_normal.groupby("Year")["SalesActual"].mean().to_dict()
        woy_mean = df_normal.groupby("WoY")["SalesActual"].mean()
        woy_factor = (woy_mean / df_normal["SalesActual"].mean()).to_dict()
        dow_mean = df_normal.groupby("DayName")["SalesActual"].mean()
        dow_factor = (dow_mean / df_normal["SalesActual"].mean()).to_dict()

        self._log("\nThree-Factor Model Factors:")
        self._log("* Annual factor (mean sales per year):", annual_factor)
        self._log("* WoY factor (mean sales per week):", woy_factor)
        self._log("* DoW factor (mean sales per day):", dow_factor)
        return annual_factor, woy_factor, dow_factor

    @staticmethod
    def _calculate_forecast_sales(
        foundation_states: Dict[str, float],
        sales_bin: float,
        event_class: Dict[str, str],
        annual_factor: Dict[int, float],
        woy_factor: Dict[int, float],
        dow_factor: Dict[str, float],
        df: pd.DataFrame,
        hist_ev: pd.DataFrame,
    ) -> None:
        df["TF_Forecast_RawSales"] = np.nan
        df["TF_Forecast_Sales"] = 0
        df["SE_Forecast_State"] = 0
        df["SE_Forecast_Sales"] = 0

        for i, row in df.iterrows():
            ev = row["EventName"]
            dow = row["DayName"]
            if pd.isna(ev):
                y = int(row["Year"])
                w = int(row["WoY"])
                af = annual_factor.get(y, np.mean(list(annual_factor.values())))
                wf = woy_factor.get(w, 1.0)
                dfac = dow_factor.get(dow, 1.0)
                tf_sales = af * wf * dfac
                state = int(np.round(tf_sales / sales_bin))
                sales_rounded = state * sales_bin
                df.at[i, "TF_Forecast_RawSales"] = int(round(tf_sales))
                df.at[i, "TF_Forecast_Sales"] = int(round(sales_rounded))
            else:
                ev_meta = hist_ev.loc[ev]
                ev_type = event_class.get(ev, "DoW_Independent")
                if ev_type == "DoW_Independent":
                    state = int(np.ceil(ev_meta["MeanState"]))
                    sales_rounded = state * sales_bin
                else:
                    foundation = foundation_states[dow]
                    delta = ev_meta["MeanDelta"]
                    state = int(max(0, np.ceil(foundation + delta)))
                    sales_rounded = state * sales_bin
                df.at[i, "SE_Forecast_State"] = state
                df.at[i, "SE_Forecast_Sales"] = int(round(sales_rounded))

        df["Intuitive_Forecast_Sales"] = (
            df["SE_Forecast_Sales"] + df["TF_Forecast_Sales"]
        )

    def assign_event_name(self, dt: datetime):
        """Return an event name like 'Christmas +2' or 'Black Friday -1' or None."""
        y = dt.year
        d = dt.date()

        # Father's Day: first Sunday of September
        sept1 = date(y, 9, 1)
        first_sunday_sept = sept1 + timedelta(days=(6 - sept1.weekday()) % 7)
        diff_fd = (d - first_sunday_sept).days
        if -1 <= diff_fd <= -1:
            if diff_fd == -0:
                return "Father's Day 0"
            return f"Father's Day {diff_fd:+d}"

        # Black Friday the last Friday of November
        nov1 = date(y, 11, 1)
        fridays = [
            nov1 + timedelta(days=i)
            for i in range(30)
            if (nov1 + timedelta(days=i)).weekday() == 4
        ]
        if len(fridays) >= 4:
            bf = fridays[-1]  # The last Friday
            diff_bf = (d - bf).days
            if -1 <= diff_bf <= 3:
                if diff_bf == 0:
                    return "Black Friday 0"
                return f"Black Friday {diff_bf:+d}"

        # Christmas cluster
        christmas = date(y, 12, 25)
        diff_x = (d - christmas).days
        if -14 <= diff_x <= 6:
            if diff_x == 0:
                return "Christmas 0"
            # diff_x = int((diff_x - int(diff_x / abs(diff_x)))/3) + int(diff_x / abs(diff_x))
            return f"Christmas {diff_x:+d}"

        # New Year
        if d == date(y, 1, 1):
            return "New Year 0"

        # Easter cluster (use dateutil.easter)
        easter_sun = easter.easter(y)
        diff_e = (d - easter_sun).days
        if -2 <= diff_e <= 1:
            # Good Friday is easter -2 -> map to "Good Friday 0" if desired
            if diff_e == -2:
                return "Good Friday 0"
            if diff_e == 0:
                return "Easter 0"
            return f"Easter {diff_e:+d}"

        # Friday Near ANZAC Day
        anzac = date(y, 4, 25)
        diff_a = (d - anzac).days
        if -3 <= diff_a <= 3:
            if diff_a == 0:
                return "ANZAC Day" + (" Shutdown" if y >= 2025 else "")
            # if d.weekday() == 4:
            #     return "Friday Near ANZAC Day"

        return None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Load and enrich
        df_sales = self._load_data()
        # Implied labour model
        reg_results, smooth_slope = self._fit_implied_labour(df_sales)
        if smooth_slope <= 1e-10:
            raise ValueError(
                "Implied labour smooth slope is too small or non-positive."
            )
        self.reg_results = reg_results
        self.smooth_slope = smooth_slope

        # Sales bin
        print("Calculating sales bin...", (self.quantum_hours, smooth_slope))
        sales_bin = int(round(self.quantum_hours / smooth_slope))
        self.sales_bin = sales_bin
        self._log("Sales bin (per quantum):", sales_bin)
        self._log("Smoothed slope:", smooth_slope)

        # Assign state
        df_sales["State"] = np.round(df_sales["SalesActual"] / sales_bin).astype(int)

        # Create a matrix of dow and state to the number of occurrences. Axis 0: State, Axis 1: DoW
        dow_state_matrix = pd.crosstab(df_sales["State"], df_sales["DayName"])
        # dow_state_matrix.to_csv(f"{self._safe_name}_dow_state_matrix.csv")
        # - for each DoW, find 3 most common States, choose the smallest state among them as foundation state
        foundation_state = {}
        for dow in dow_state_matrix.columns:
            state_counts = dow_state_matrix[dow]
            top_states = (
                state_counts.sort_values(ascending=False).head(3).index.tolist()
            )
            min_state = min(top_states)
            if dow == "Sunday":
                min_state += 1  # Sunday bias adjustment

            foundation_state[dow] = min_state
        self.foundation_state = foundation_state
        # print foundation state with sorted DoW: Monday to Sunday
        self._log(
            "\nFoundation state (per DoW):",
            {
                dow: foundation_state[dow]
                for dow in [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
            },
        )

        # Special events + Event names
        df_sales, event_class = self._mark_special_and_events(df_sales, sales_bin)
        self.event_class = event_class
        # df_sales.to_csv("df_sales_enriched.csv", index=False)
        # Three-factor from normal days
        annual_factor, woy_factor, dow_factor = self._three_factor(df_sales)

        # Forecast frame for target period
        future_dates = pd.date_range(self.forecast_start, self.forecast_end, freq="D")
        df_forecast = pd.DataFrame({"Date": future_dates})
        df_forecast["Year"] = df_forecast["Date"].dt.year
        df_forecast["WoY"] = get_week_of_year(df_forecast["Date"])
        df_forecast["DayName"] = df_forecast["Date"].dt.day_name()

        # Historical repeated special events summary
        hist_ev = (
            df_sales[df_sales["EventName"] != "Unclassified"]
            .assign(
                Delta=lambda x: x["State"] - x["DayName"].map(self.foundation_state)
            )
            .groupby("EventName")
            .agg(
                Years=("Year", lambda s: sorted(s.unique())),
                YearsCount=("Year", "nunique"),
                MeanState=("State", "mean"),
                MeanDelta=("Delta", "mean"),
                ExampleDays=("Date", lambda s: list(s.dt.strftime("%Y-%m-%d")[:5])),
            )
            .reset_index()
            .set_index("EventName")
        )
        self.hist_ev = hist_ev
        # hist_ev.to_csv(f"{self._safe_name}_special_events.csv")
        if not hist_ev.empty:
            self._log("Repeated special events detected:")
            self._log(hist_ev[["YearsCount", "MeanState", "MeanDelta"]])

        # Event tagging in forecast only if repeated historically
        df_forecast["EventName"] = df_forecast["Date"].apply(self.assign_event_name)
        df_forecast["EventName"] = df_forecast["EventName"].where(
            df_forecast["EventName"].isin(hist_ev.index), None
        )

        # Intuitive forecast per day
        self._calculate_forecast_sales(
            self.foundation_state,
            sales_bin,
            event_class,
            annual_factor,
            woy_factor,
            dow_factor,
            df_forecast,
            hist_ev,
        )

        # Random Forest forecast
        df_forecast["RandomForest_Forecast"], actual_SalesForecast = Predict().predict(
            df_sales=df_sales[["Date", "SalesActual"]]
            .copy()
            .rename(columns={"SalesActual": "Sales"}),
            start_forecast=self.forecast_start,
            end_forecast=self.forecast_end,
        )
        df_forecast.loc[:, "RandomForest_Forecast"] = (
            df_forecast["RandomForest_Forecast"].round().astype(int)
        )
        actual_SalesForecast = actual_SalesForecast.round().astype(int)

        # Hybrid blend
        df_forecast["Hybrid_Forecast_Sales"] = (
            (
                self.blend_weight * df_forecast["Intuitive_Forecast_Sales"]
                + (1 - self.blend_weight) * df_forecast["RandomForest_Forecast"]
            )
            .round()
            .astype(int)
        )

        df_forecast["Hybrid_State"] = np.round(
            df_forecast["Hybrid_Forecast_Sales"] / sales_bin
        ).astype(int)
        df_forecast["Hybrid_Sales_Rounded"] = df_forecast["Hybrid_State"] * sales_bin
        df_forecast["Hybrid_Labour"] = df_forecast.apply(
            lambda r: self.smooth_slope * r["Hybrid_Sales_Rounded"]
            + self.reg_results[r["DayName"]]["intercept_smoothed"],
            axis=1,
        )

        # History + predicted for history
        df_sales_result = df_sales[
            [
                "Date",
                "Year",
                "WoY",
                "DayName",
                "SalesActual",
                "SalesForecast",
                "HoursActual",
                "HoursForecast",
                "EventName",
            ]
        ].copy()
        df_sales_result["EventName"] = df_sales_result["EventName"].where(
            df_sales_result["EventName"].isin(hist_ev.index), None
        )
        self._calculate_forecast_sales(
            self.foundation_state,
            sales_bin,
            event_class,
            annual_factor,
            woy_factor,
            dow_factor,
            df_sales_result,
            hist_ev,
        )
        df_sales_result["RandomForest_Forecast"] = actual_SalesForecast
        df_sales_result["Hybrid_Forecast_Sales"] = (
            (
                self.blend_weight * df_sales_result["Intuitive_Forecast_Sales"]
                + (1 - self.blend_weight) * df_sales_result["RandomForest_Forecast"]
            )
            .round()
            .astype(int)
        )
        df_sales_result["Hybrid_State"] = np.round(
            df_sales_result["Hybrid_Forecast_Sales"] / sales_bin
        ).astype(int)
        df_sales_result["Hybrid_Sales_Rounded"] = (
            df_sales_result["Hybrid_State"] * sales_bin
        )

        # Save outputs
        # df_forecast.to_csv(f"{self._safe_name}_quantum_forecast.csv", index=False)
        df_forecast_result = df_forecast[
            [
                "Date",
                "Year",
                "WoY",
                "DayName",
                "EventName",
                "TF_Forecast_RawSales",
                "TF_Forecast_Sales",
                "SE_Forecast_State",
                "SE_Forecast_Sales",
                "Intuitive_Forecast_Sales",
                "RandomForest_Forecast",
                "Hybrid_Forecast_Sales",
                "Hybrid_State",
                "Hybrid_Sales_Rounded",
            ]
        ].copy()

        df_full = pd.concat([df_sales_result, df_forecast_result], ignore_index=True)
        df_full["Hybrid_Forecast_Labour"] = df_full.apply(
            lambda r: self.smooth_slope * r["Hybrid_Sales_Rounded"]
            + self.reg_results[r["DayName"]]["intercept_smoothed"],
            axis=1,
        ).round(1)
        # df_full.to_csv(
        #     f"{self._safe_name}_full_quantum_forecast_with_history.csv", index=False
        # )
        self.df_full_forecast = df_full

        output_data = {
            "Data": df_full.to_dict(orient="records"),
            "ForecastStart": self.forecast_start.strftime("%Y-%m-%d"),
            "ForecastEnd": self.forecast_end.strftime("%Y-%m-%d"),
            "FoundationState": self.foundation_state,
            "SalesBin": self.sales_bin,
            "SmoothSlope": self.smooth_slope,
            "SpecialEvents": self.hist_ev[["ExampleDays"]]
            .reset_index()
            .to_dict(orient="records"),
        }
        self.output_data = output_data
        # with open(f"{self._safe_name}_quantum_forecast_output.json", "w") as f:
        #     json.dump(output_data, f, indent=4, default=str)

        # Error analysis
        df_error = df_sales_result.dropna(
            subset=["SalesActual"]
        ).copy()  # only historical data with actual sales
        df_error["O_Error"] = df_error["SalesActual"] - df_error["SalesForecast"]
        df_error["RF_Error"] = (
            df_error["SalesActual"] - df_error["RandomForest_Forecast"]
        )
        df_error["Intuitive_Error"] = (
            df_error["SalesActual"] - df_error["Intuitive_Forecast_Sales"]
        )
        print(
            "Original Forecast RMSE:",
            mean_squared_error(df_error["SalesActual"], df_error["SalesForecast"]),
        )
        print(
            "Random Forest Forecast RMSE:",
            mean_squared_error(
                df_error["SalesActual"], df_error["RandomForest_Forecast"]
            ),
        )
        print(
            "Intuitive Forecast RMSE:",
            mean_squared_error(
                df_error["SalesActual"], df_error["Intuitive_Forecast_Sales"]
            ),
        )
        # - draw error line plot
        # -- df_error group by month then average the errors
        df_error["Month"] = df_error["Date"].dt.strftime("%Y-%m")
        df_error = (
            df_error.groupby("Month")[["O_Error", "RF_Error", "Intuitive_Error"]]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(12, 6))
        plt.plot(
            df_error["Month"],
            df_error["O_Error"],
            label="Original Forecast Error",
            alpha=0.7,
        )
        plt.plot(
            df_error["Month"],
            df_error["RF_Error"],
            label="Random Forest Forecast Error",
            alpha=0.7,
        )
        plt.plot(
            df_error["Month"],
            df_error["Intuitive_Error"],
            label="Intuitive Forecast Error",
            alpha=0.7,
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("Month", rotation=90)
        plt.ylabel("Forecast Error")
        plt.title(f"Forecast Error Comparison for {self.name}")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"{self._safe_name}_forecast_error_comparison.png")
        plt.close()

        # self._log(
        #     "Forecast generated -> quantum_forecast.csv & full_quantum_forecast_with_history.csv"
        # )

        # Store for access
        self.df_sales = df_sales
        self.df_forecast = df_forecast
        return df_sales, df_forecast


if __name__ == "__main__":

    # 1) Run the quantum forecast pipeline
    q = QuantumForecast(
        # name = "Mecca Castle Towers",
        name="Mecca Double Bay",
        # name="Mecca Parramatta",
        # input_json="new_data/sales_data_Mecca_Castle_Towers.json",
        input_json="new_data/sales_data_Mecca_Double_Bay.json",
        # input_json="new_data/sales_data_Mecca_Parramatta.json",
        forecast_end="2025-12-31",
        # forecast_end="2024-12-31",
    )
    q.run()
