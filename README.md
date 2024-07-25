# Description:
## Objective:
Built based on OOP, this portfolio generator introduces Modern Portfolio Theory - Mean Variance Optimization to optimize asset allocation for maximum return at minimal risk, utilizes ***Fama-French Five-Factor Model*** to predict expected returns using Linear Regressions, and leverages ***OAS*** method to apply shrinkage on the covariance matrix.

## Details:
1. Mean Variance Optimization: Use the function `minimize` to find the weight of each asset that yields the best performance (variance, return, sharpe ratio, utility score, etc)

2. Fama-French Five-Factor Model (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): Predict expected return using the linear regression of alpha + beta's * factor's

    a. **Rm-Rf** is the excess return on the market, value-weight return of all CRSP firms incorporated in the US and listed on the NYSE, AMEX, or NASDAQ that have a CRSP share code of 10 or 11 at the beginning of month t, good shares and price data at the beginning of t, and good return data for t minus the one-month Treasury bill rate (from Ibbotson Associates)

    b. **SMB (Small Minus Big)** is the average return on the nine small stock portfolios minus the average return on the nine big stock portfolios

    c. **HML (High Minus Low)** is the average return on the two value portfolios minus the average return on the two growth portfolios

    d. **RMW (Robust Minus Weak)** is the average return on the two robust operating profitability portfolios minus the average return on the two weak operating profitability portfolios

    e. **CMA (Conservative Minus Aggressive)** is the average return on the two conservative investment portfolios minus the average return on the two aggressive investment portfolios

3. OAS method: Calculate the covariance matrix
