# Price Impact Study

A script that takes orderbook data and plots the price impact of buyer initiated trades against bins of volume. Inspired by "Single Curve Collapse of the Price Impact Function for the New York Stock Exchange” by Lillo, Farmer, and Mantegna.



### Summary and Structure:

LOBSTERdata was used to obtain free limit orderbook and messagebook datasets. The orderbook contains the best bid/ask prices and size while the messagebook contains events causing an update to the orderbook (new order submissions, cancellations, executions, etc.) as well as a timestamp for every event. Every row in the orderbook represents the state of the book after an event in the messagebook (same row number).

The data is read in and rows are converted into class objects corresponding to either the order or messagebook and put into a list.
We then take only data relating to buy events (buyer initiated execution of visibile or hidden limit orders) and aggregate trades with the same timestamps. One trade would be recorded accross multiple rows if it could not be filled at one price, in these cases the rows would be combined resulting in one row containing the entire transactions volume and final state of the orderbook.

Volume is then normalized by dividing by the average transaction volume in order to enable comparing between stocks.

Trades are then binned by this volume multiple and the price impact of each trade (change in ask price) in a bin is used to obtain an average impact.

This data is then plotted. Code for fitting an exponential function is provided but not used at the bottom, since, unfortunately the datasets were insufficient for such analysis. While a positive correlation can be seen in the graphs between volume and price impact, after extracting the relevant data, we were left with a limited number of relevant transaction. This in turn led to a large degree of variance in measuring price impact in bins that were sparsly populated.
