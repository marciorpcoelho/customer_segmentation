# Customer Segmentation Preliminary Analysis - CA

This the description of the first and preliminary steps in Customer Segmentation for only CA clients.

###Selected DBs and Tables:

BI_CA - VHE_Sales - Sales of New/Used Vehicles;  
BI_CA - PSE_Sales - After-Sales Services;  
BI_CA - SLR_PaidTransactions - Transactions already paid;   
BI_CA - SLR_OpenTransactions - Transactions yet to pay;


###Preprocessing of Data
**1<sup>st</sup> step: Removal of useless information**  
Some columns provide little to no information, such as the Registration_Number or the NLR_Code (same for all DBs). As such, they can be safely discarded.


**2<sup>nd</sup> step: NULL Analysis**  
In this step null values are counted and removed in the columns which can not be imputed. Examples of these are fiscalNumber, SLR_Account, etc.  
PaidTransactions: 43% of records without fiscalNumber and 43% without Centre Information - Removed.  
OpenTransactions: 4.7% of records without section Section Information - Removed.


**3<sup>rd</sup> step: Erroneous data**  
Some of the information left had erroneous values such as 0 or 1 for the Client Account Number, 999999990 or 0 for fiscal Number. These odd values were probably caused by migration errors or non-available data. As such, these are also removed.
In total, 7225 entries were removed from PaidTransactions, and 47 from OpenTransactions. **Also removed, were clients who had a total sum of 0 in their respective transactions. A total of 1773 entries were removed.**


**4<sup>th</sup> step: Data Engineering**  
The VHE_Sales provides the number of cars bought by each client. As such, it can be converted to a single column referring to the number of bought vehicles.





###Exploratory Analysis
