`pip install --editable .[all]`
**Note**: all option will install all the dependencies which might not be needed for your workflow, alternatively following options are available to install minimal dependencies as per need:  

`pip install obsei\[source\]`: To install dependencies related to all observers.  

`pip install obsei\[sink\]`: To install dependencies related to all informers.  

`pip install obsei\[analyzer\]`: To install dependencies related to all analyzers, it will install pytorch as well.  

`pip install obsei[twitter-api]`: To install dependencies related to Twitter observer.  

`pip install obsei[google-play-scraper]`: To install dependencies related to Play Store review scrapper observer.  

`pip install obsei[google-play-api]`: To install dependencies related to Google official play store review API based observer.  

`pip install obsei[app-store-scraper]`: To install dependencies related to Apple App Store review scrapper observer.  

`pip install obsei[reddit-scraper]`: To install dependencies related to Reddit post and comment scrapper observer.  

`pip install obsei[reddit-api]`: To install dependencies related to Reddit official api based observer.  

`pip install obsei[pandas]`: To install dependencies related to TSV/CSV/Pandas based observer and informer.  

`pip install obsei[google-news-scraper]`: To install dependencies related to Google news scrapper observer.  

`pip install obsei[facebook-api]`: To install dependencies related to Facebook official page post and comments api based observer.  

`pip install obsei[atlassian-api]`: To install dependencies related to Jira official api based informer.  

`pip install obsei[elasticsearch]`: To install dependencies related to elasticsearch informer.   

`pip install obsei[slack-api]`:To install dependencies related to Slack official api based informer.  

You can also mix multiple dependencies together in single installation command. For example to install dependencies Twitter observer, all analyzer, and Slack informer use following command:  

pip install obsei[dependecy1, dependecy2, dependecy3, ...]  

For instance: 
`pip install obsei[twitter-api, analyzer, slack-api]`