# Customer Segmentation Preliminary Analysis - CA

This the description of the first and preliminary steps in Customer Segmentation for only CA clients.

###Selected DBs and Tables:

BI_CA - VHE_Sales - Sales of New/Used Vehicles;  
BI_CA - PSE_Sales - After-Sales Services;  
BI_CA - SLR_PaidTransactions - Transactions already paid;   
BI_CA - SLR_OpenTransactions - Transactions yet to pay;


###Preprocessing of Data
**1<sup>st</sup> step: Removal of useless information**   
For the initial exploration, where the focus is only in the BI_CA, the columns NLR_Code can be safely disregarded, as they are the same for all DB's.


**2<sup>nd</sup> step: NULL Analysis**  
In this step null values are counted and removed in the columns which can not be imputed. Examples of these are fiscalNumber, SLR_Account, etc.  
PaidTransactions: 43% of records without fiscalNumber and 43% without Centre Information - Removed.  
OpenTransactions: 4.7% of records without section Section Information - Removed.


**3<sup>rd</sup> step: Erroneous data**  
Some of the information left had erroneous values such as 0 or 1 for the Client Account Number and 999999990 or 0 for fiscal Number. These odd values were probably caused by migration errors or non-available data. As such, these are also removed.
In total, 7225 entries were removed from PaidTransactions, and 47 from OpenTransactions. **Also removed, were clients who had a total sum of 0 in their respective transactions. A total of 1773 entries were removed.**


**4<sup>th</sup> step: Data Engineering**  
The VHE_Sales provides the number of cars bought by each client. As such, it can be converted to a single column referring to the number of bought vehicles, counting on the number of unique Registration_Number per client. Also added is their respective total_invoice, separated by new and used vehicles.  
Also created, are the total per transaction type separated between Paid and Open transactions.

**5<sup>th</sup> step: Time period selection**  
Since all the databases have their own unique time interval, the focus was on the common period across all, which is the full year of 2017.


###Exploratory Analysis
Transaction Value Analysis:

![](./output/1_transaction_distribution_2017.png)

First analysis shows the expected decreases in number of transactions, as their value increase. Also noticeable is the higher cost of new vehicles when compared to used ones.
Also relevant are the considerable high frequency of low cost transactions associated with new vehicles - **Need to look into this.** 