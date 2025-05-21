#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pylab
import numpy as np


class orderbook_row(object):
    """
    Represents a row in an orderbook dataset. Each row contains a best bid and
    ask price as well as the size at each.
    """

    def __init__(self, ask_price, ask_size, bid_price, bid_size, row_num):
        """
        Args:
            ask_price (int): Best Ask dollar price times 10000. i.e. stock
                price of $93.74 would be 937400
            ask_size (int): Best Ask volume
            bid_price (int): Best Bid dollar price times 10000
            bid_size (int): Best Bid volume
        """

        self.ask_price = ask_price
        self.ask_size = ask_size
        self.bid_price = bid_price
        self.bid_size = bid_size
        self.row_num = row_num
        self.time = None

    def get_ask_price(self):
        return self.ask_price

    def get_ask_size(self):
        return self.ask_size

    def get_bid_size(self):
        return self.bid_size

    def get_bid_price(self):
        return self.bid_price

    def get_row_num(self):
        return self.row_num

    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time

    def __str__(self):
        return f"[{self.ask_price}, {self.ask_size}, {self.bid_price}, \
{self.bid_size}, {self.row_num}, {self.time}]"

    def __repr__(self):
        return f"[{self.ask_price}, {self.ask_size}, {self.bid_price}, \
{self.bid_size}, {self.row_num}, {self.time}]"


class message_row(object):
    """
    Represents a row in the message dataset. Each row contains a timestamp,
    trade type, order ID, size (number of shares), and price.
    """
    def __init__(self, time, t_type, order_id, size, price, direction, row_num):
        """
        Args:
            time (float): timestamp of an event
            t_type (int): [1-7] representing 1 of 7 trade events
            order_id (int):  representing an order reference number
            size (int): number of shares
            price (int): price in dollars times 10000
            direction (int): 1 for a buy limit order or -1 for a sell l.o.,
            execution of a buy l.o.(1) corresponds to a seller initiated trade
       """
        self.time = time
        self.t_type = t_type
        self.order_id = order_id
        self.size = size
        self.price = price
        self.direction = direction
        self.row_num = row_num

    def get_time(self):
        return self.time

    def get_t_type(self):
        return self.t_type

    def get_order_id(self):
        return self.order_id

    def get_size(self):
        return self.size

    def get_price(self):
        return self.price

    def get_direction(self):
        return self.direction

    def get_row_num(self):
        return self.row_num

    def __str__(self):
        return f"[{self.time}, {self.t_type}, {self.order_id}, \
{self.size}, {self.price}, {self.get_direction}, {self.row_num}]"

    def __repr__(self):
        return f"[{self.time}, {self.t_type}, {self.order_id}, \
{self.size}, {self.price}, {self.direction}, {self.row_num}]"


def read_orderbook(orderbook_path):
    """
    Parameters
    ----------
    orderbook_path : string of the filepath to data

    Returns
    -------
    orderbook : list of orderbook_row objects
    """
    orderbook = []
    try:
        with open(orderbook_path, 'r') as file:
            orderbook_raw = csv.reader(file)
            orderbook_list = list(orderbook_raw)
            row_counter = 0

            for row in orderbook_list:                     # Each row is a list

                orderbook.append(orderbook_row(int(row[0]), int(row[1]),
                                               int(row[2]), int(row[3]),
                                               row_counter))
                row_counter += 1

        return orderbook

    except FileNotFoundError:
        raise Exception(f"Error: File not found at '{orderbook_path}'")


def read_message(message_path):
    """
    Parameters
    ----------
    message_path : string of the filepath to data

    Returns
    -------
    messages : list of message_row objects
    """
    messages = []
    try:
        with open(message_path, 'r') as file:
            message_raw = csv.reader(file)
            message_list = list(message_raw)
            row_counter = 0

            for row in message_list:

                messages.append(message_row(float(row[0]), int(row[1]),
                                            int(row[2]), int(row[3]), int(row[4]),
                                            int(row[5]), row_counter))
                row_counter += 1

        return (messages)

    except FileNotFoundError:
        raise Exception(f"Error: File not found at '{message_path}'")


# Obtaining just the transaction data
def transactions_only(message_data, book_data):
    """
    Parameters
    ----------
    message_data : list of message_row objects
    book_data : list of orderbook_row objects

    Returns
    -------
    A tuple containing:
    message_trans : list of message_rows corresponding to transactions only
    book_trans : list of orderbook_rows corresponding to transactions only
    """
    message_trans = []
    for row in message_data:
        t_type = row.get_t_type()
        if t_type == 4 or t_type == 5:       #type 4 and 5 are executed orders
            direction = row.get_direction()

            if direction == -1:               #-1 for buy orders only
                message_trans.append(row)

    # Taking the row numbers from transactions
    desired_rows = []
    for row in message_trans:
        desired_rows.append(row.get_row_num())

    # Using those row numbers to take transactions from orderbook
    book_trans = []
    for num in desired_rows:
        book_trans.append(book_data[num])

    return (message_trans, book_trans)


def get_trans_vol(messagebook):
    """
    Parameters
    ----------
    messagebook : list of orderbook_row objects

    Returns
    -------
    trans_vol : list of trade volume
    """
    trans_vol = []
    for row in messagebook:
        trans_vol.append(row.get_size())

    return trans_vol


def avg_prices(book_data):
    """
    Calculates the avg ask price over all the data.
    Never actually used.
    """
    avg_price = 0
    for i in range(len(book_data)):

        avg_price += book_data[i].get_ask_price()

    avg_price = avg_price / len(book_data)

    return avg_price


def agg_trans(msg_trans, order_trans):
    """
    Parameters
    ----------
    msg_trans : List of message_rows corresponding to transactions only
    order_trans : List of orderbook_rows corresponding to transactions only

    Returns
    -------
    agg_msg : Dictionary of time:volume
    agg_order : Dictionary of time:ask_price
    """
    agg_msg = {}
    for i in range(len(msg_trans)):

        key = msg_trans[i].get_time()
        value = msg_trans[i].get_size()

        if key not in agg_msg.keys():
            agg_msg[key] = value
        else:
            agg_msg[key] = agg_msg[key] + value

    for j in range(len(msg_trans)):
        time = msg_trans[j].get_time()
        order_trans[j].set_time(time)

    agg_order = {}
    for order in order_trans:
        time = order.get_time()
        price = order.get_ask_price()

        agg_order[time] = price

    return (agg_msg, agg_order)


def price_change(trans_order):
    """
    Parameters
    ----------
    trans_order : List of orderbook transactions

    Returns
    -------
    changes : List of price change between each transaction
    """
    changes = []
    for i in range(len(trans_order)-1):
        change = abs(trans_order[i+1] - trans_order[i])
        changes.append(change)
    return changes


def vol_ratio(vol_data, avg_vol):
    """
    Parameters
    ----------
    vol_data : List of volume per each transaction to be transformed
    avg_vol : Float of mean volume to use in calculating vol. ratio

    Returns
    -------
    vol_data : List of each trade volume as a ratio of mean trade volume
    """
    for i in range(len(vol_data)):
        if vol_data[i] > 0:
            vol_data[i] = vol_data[i]/avg_vol

    return vol_data


def bin_and_average(data, bin_size):
    """
    Bins the keys of a dictionary and averages the values for each bin.

    Args:
        data (dict): A dictionary with numerical keys and values.
        bin_size (int): The size of each bin.

    Returns:
        list: A list containing the the max edge value of each bin.
        list: A list containing the average values of each bin.
    """
    binned_data = {}
    for key, value in data.items():
        bin_start = (key // bin_size) * bin_size
        bin_end = bin_start + bin_size - 1
        bin_range = (bin_start, bin_end)

        if bin_range not in binned_data:
            binned_data[bin_range] = {"sum": 0, "count": 0}

        binned_data[bin_range]["sum"] += value
        binned_data[bin_range]["count"] += 1

    averaged_data = {
        bin_range: bin_data["sum"] / bin_data["count"]
        for bin_range, bin_data in binned_data.items()
    }
    keys = list(averaged_data.keys())

    bin_edge = []
    for tuples in keys:
        bin_edge.append(tuples[1])

    price_change = list(averaged_data.values())

    return bin_edge, price_change


def calc_impact_and_graph(orderbooks, messagebooks):
    """
    Function for calculating the price impact of a trade given an order size.
    Order sizes are binned as multiples of the average volume over the
    dataset. The mean price impact for each bin will then be used in the graph.

    Parameters
    ----------
    orderbooks : list of filepaths to csv file containing orderbooks data
    messagebooks : list of filepaths to csv file containing messagebooks data
    """

    for i in range(len(orderbooks)):
        name = orderbooks[i]     #For labeling graph later
        message_data = read_message(messagebooks[i])
        book_data = read_orderbook(orderbooks[i])

        transactions_data = transactions_only(message_data, book_data)
        trans_message = transactions_data[0]
        trans_order = transactions_data[1]

        transaction_vol = get_trans_vol(trans_message)

        avg_vol = 0
        for x in transaction_vol:
            avg_vol += x

        avg_vol = avg_vol / len(transaction_vol)

        aggregated_trans = agg_trans(trans_message, trans_order)

        agg_trans_dict = aggregated_trans[0]
        trans_vol = list(agg_trans_dict.values())

    # Delete the first elements as it won't correspond to a change in price
        trans_vol.pop(0)

        trans_ord_clean = list(aggregated_trans[1].values())

        price_changes = price_change(trans_ord_clean)
        print(price_changes)

    # Prices in data were *10000 so we set it back to normal here
        price_changes = (np.array(price_changes)/10000)

        price_changes = np.log(price_changes)

        volratio_clean = vol_ratio(trans_vol, avg_vol)


    # Making a dictionary to create bins based on vol ratio with mean values for
    # each bin
        data_dict = {}
        for i in range(len(price_changes)):
            key = volratio_clean[i]
            val = price_changes[i]
            data_dict[key] = val

        bins, price_change_scram = bin_and_average(data_dict, 3)

    # The returned lists aren't in order so the following lines reorder the bins
    # using a dictionary to keep track of corresponding price_change(y-values)
        temp_dict = {}
        for j in range(len(bins)):
            key = bins[j]
            val = price_change_scram[j]
            temp_dict[key] = val

        bins.sort()

        price_change_final = []

        for i in bins:
            change_val = temp_dict[i]
            price_change_final.append(change_val)

        name = name[0:4]
        pylab.plot(bins, price_change_final)
        # Cutoff chosen at 25 times avg. volume as bins are nearly empty after that
        pylab.xlim(1, 25)
        pylab.xlabel('Multiple of Avg. Order Size')
        pylab.ylabel('Price Shift')
        pylab.title("Price Impact " + name)
        pylab.show()


if __name__ == "__main__":

    orderbooks_list = ['AAPL_2012-06-21_34200000_57600000_orderbook_1.csv',
                       'MSFT_2012-06-21_34200000_57600000_orderbook_1.csv',
                       'INTC_2012-06-21_34200000_57600000_orderbook_1.csv']

    messagebooks_list = ['AAPL_2012-06-21_34200000_57600000_message_1.csv',
                         'MSFT_2012-06-21_34200000_57600000_message_1.csv',
                         'INTC_2012-06-21_34200000_57600000_message_1.csv']

    calc_impact_and_graph(orderbooks_list, messagebooks_list)






"""
The lines below could be used to fit an exponential function if inserted in
calc_impact_and_graph(), however, due to the very limited datasets there isn't
much value in doing so.
"""

# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt

# x_data =  np.array(bins)
# y_data = np.array(price_change_final)

# x_data = x_data[0:10]
# y_data = y_data[0:10]

# def exp_func(x,a, b):
#     return a*np.exp(b*x)

# poptimal, pcov = curve_fit(exp_func, x_data, y_data)

# a_opt, b_opt = poptimal

# y_fitted = exp_func(x_data, a_opt, b_opt)

# plt.scatter(x_data, y_data, label="Original Data")
# plt.plot(x_data, y_fitted, label=f"Fitted Function: y = {a_opt:.2f} * exp({b_opt:.2f}x)", color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
