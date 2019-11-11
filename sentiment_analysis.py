import re 
from textblob import TextBlob 
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import classifiers
from nltk.tokenize import sent_tokenize

training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
('Has made frequent errors that are harmful to business operations.', 'neg'),
('The supervisor/department head has received numerous complaints about the quality of work.', 'neg'),
('The quality of work produced is unacceptable.', 'neg'),
('Does not complete required paperwork.', 'neg'),
('Is not as careful in checking work product for errors as he/she could be.', 'neg'),
('Tends to miss small errors in work product.', 'neg'),
('Required paperwork is completed late or is only partially complete.', 'neg'),
('Often calls in to work without prior approval, resulting in excessive unscheduled absences.', 'neg'),
('Leaves the work area unattended to run personal errands.', 'neg'),
('Is frequently late to work', 'neg'),
('Frequently leaves work early.', 'neg'),
('Occasionally calls in to work without prior approval, resulting in unscheduled absences.', 'neg'),
('Occasionally arrives late to work.', 'neg'),
('Sometimes does not make sure all work is completed before leaving for the day.', 'neg'),
('Occasionally leaves work early.', 'neg'),
('Projects an attitude of superiority that turns off other employees.', 'neg'),
('Not cooperative and frequently criticizes others.', 'neg'),
('Displays excessive negativity when working with others.', 'neg'),
('Displays occasional negativity when working with others.', 'neg'),
('Rarely offers to assist others in the office.', 'neg'),
('Makes negative comments that affect working relationships with others.', 'neg'),
('Usually needs direct supervision, even for mundane and everyday tasks.', 'neg'),
('Is not able to think independently or to deal with unexpected occurrences.', 'neg'),
('Gets flustered in unusual situations.', 'neg'),
('Does not always make the best decisions to fit the situation.', 'neg'),
('Reports, forms, memos and correspondence are often completed late or not at all.', 'neg'),
('Uses a condescending tone when talking to others in the office.', 'neg'),
('The supervisor/department head has received a few complaints about contradictory or bad information being given out by the employee.', 'neg'),
('Phone messages are often unclear or incomplete.', 'neg'),
('Frequently comes to the wrong conclusions and assumes things.', 'neg'),
('Did not make sure that all subordinates were productive at all times, which is a daily requirement of this job.', 'neg'),
('Needs to develop analytical skills necessary to weigh options and choose the best way to deal with situations.', 'neg'),
('Spends too much time focusing on less important aspects of daily job.', 'neg'),
('Frequently rude and impolite.', 'neg'),
('Demonstrates poor customer relations skills.', 'neg'),
('Frequently carries on personal conversations in person or on the phone while clients and customers wait.', 'neg'),
('Gets annoyed with clients who ask too many questions.', 'neg'),
('Frequently forgets to follow through on customer requests.', 'neg'),
('Has destroyed equipment through misuse during this rating period.', 'neg'),
('Wastes supplies.', 'neg'),
('Deleted required software in error.', 'neg'),
('Never services equipment.', 'neg'),
('Doesn\'t heed warning messages on equipment.', 'neg'),
('Sometimes forgets to turn equipment off at the end of the day.', 'neg'),
('Doesn\'t always get equipment serviced as recommended by the manufacturer.', 'neg'),
('Work projects have suffered from lack of follow-through.', 'neg'),
('Important documentation for projects has been lost or destroyed erroneously.', 'neg'),
('Does not plan ahead to meet work deadlines.', 'neg'),
('Does not keep supervisor informed of potential problems as they arise.', 'neg'),
('Project plans are poorly designed.', 'neg'),
('Project plans are not carried out as assigned or on time.', 'neg'),
('Dictates to others rather than involving them in the decision making.', 'neg'),
('Has reduced subordinates to tears.', 'neg'),
('Yells and screams at subordinates.', 'neg'),
('Assumes others should know what to do and how to do it with little or no training.', 'neg'),
('Frequently becomes impatient when things aren\'t done their way.', 'neg'),
('Had one unrated Performance Planning and Review rating in this rating year.', 'neg'),
('Did not conduct timely planning sessions on all subordinates.', 'neg'),
('Although planning sessions were completed, they were not completed within Civil Service mandated timelines.', 'neg'),
('Did not meet personally with the employee to go over appraisals.', 'neg'),
('Does not require constant supervision.','pos'),
('Error rate is acceptable, and all work is completed timely.','pos'),
('Forms and required paperwork are completed on time with minimal errors.','pos'),
('Managers and co-workers have commented on high levels of accuracy and work productivity.','pos'),
('Takes pride in work and strives to improve work performance.','pos'),
('All memos, reports, forms and correspondence are completed on time with no errors.','pos'),
('Has less than a 1% error rate on work product.','pos'),
('Accuracy is excellent.','pos'),
('Quantity of work produced is outstanding.','pos'),
('Consistently arrives to work on time.','pos'),
('Makes sure work area is covered at all times.','pos'),
('Has had no unscheduled absences, except for documented emergencies.','pos'),
('Has a good attendance record.','pos'),
('Can always be counted on to work overtime when necessary without complaint.','pos'),
('Always at work and on time.','pos'),
('Never misses work without prior approval and appropriate notification.','pos'),
('Has had no unscheduled absences during the rating period.','pos'),
('Is usually able to answer customer questions.','pos'),
('Maintains good working relationships with coworkers.','pos'),
('Demonstrates “team player” behavior views individual success as imperative to group success.','pos'),
('Direct, straightforward, honest and polite.','pos'),
('Always cordial and willing to help coworkers, students, and clients.','pos'),
('Enthusiastic, energetic and displays positive behavior.','pos'),
('Usually adjusts well to changes in the work place.','pos'),
('Maintains good customer service relations, even under stress.','pos'),
('Looks for ways to streamline procedures to improve efficiency and customer service.','pos'),
('Sets priorities and adjusts them as needed when unexpected situations arise.','pos'),
('Adapted to new systems and processes well and seeks out training to enhance knowledge, skills and abilities.','pos'),
('Always seems to know when to ask questions and when to seek guidance.','pos'),
('Takes messages, writes correspondence, deals with customers and coworkers with sufficient attention to detail.','pos'),
('Reports are accurate and well written using proper grammar and punctuation.','pos'),
('Students and coworkers feel comfortable coming to this employee with questions and comments.','pos'),
('Comes to supervisor/department head with any questions that employee does not know off-hand','pos'),
('Always asks questions and seeks guidance when not sure of what to do.','pos'),
('Demonstrates excellent oral and written communication skills.','pos'),
('Often offers workable solutions to problems.','pos'),
('Uses good judgment in solving problems and working with others.','pos'),
('Uses PPR ratings in making decisions related to new hires, promotions and merit increases.','pos'),
('Can zero in on the cause of problems and offer creative solutions.','pos'),
('Displays strong analytical skills.','pos'),
('Always offers ideas to solve problems based on good information and sound judgment.','pos'),
('Displays initiative and enthusiasm during everyday work.','pos'),
('Conducts research or seeks counsel of experts to gather information needed in making actual decisions.','pos'),
('Usually maintains a competent and professional demeanor in dealing with clients and the public.','pos'),
('Courteous and knowledgeable.','pos'),
('Tries to be helpful.','pos'),
('Answers all questions promptly and accurately.','pos'),
('Forwards any complaints or problems to supervisor immediately.','pos'),
('Always follows through and finds the answers to any questions and reports back to the customer promptly.','pos'),
('Employee has received numerous letters of commendation for excellent customer service.','pos'),
('Takes good care of equipment and uses supplies efficiently.','pos'),
('Turns off and secures all equipment at the end of the shift.','pos'),
('Quickly learns new software programs.','pos'),
('Uses queries and reports to maximize efficiency in the office and find errors.','pos'),
('Is able to troubleshoot and solves all work related problems quickly and efficiently.','pos'),
('Reports problems immediately if to the appropriate personnel.','pos'),
('Prepares project plans on time and in sufficient detail.','pos'),
('End of year statements are complete and accurate.','pos'),
('Maintains and monitors progress of project plan in order to stay on target.','pos'),
('Gets the most out of scarce resources.','pos'),
('Projects normally are within budget and are well planned.','pos'),
('Anticipates problems before they occur.','pos'),
('Provides meaningful information to decision makers that helps in the preparation and implementation of projects.','pos'),
('Plans projects and carries them out so that projects are completed ahead of schedule and under budget.','pos'),
('Draws on the knowledge and skills of others.','pos'),
('Available when needed and has an open door policy for subordinates.','pos'),
('Assigns work fairly and resolves disputes and grievances of subordinates fairly.','pos'),
('Very supportive of coworkers and subordinates attempts at improvement.','pos'),
('Sets an example for subordinates in following departmental and university policy and procedures.','pos'),
('Outstanding ability to explain and teach.','pos'),
('Inspires others to do better.','pos'),
('All PPR\'s were completed by the anniversary dates of all subordinates.','pos'),
('Works with employees in setting mutual goals.','pos'),
('Makes an effort to counsel employees and document performance (both positive and negative) throughout the year.','pos'),
('Maintains a supervisor file that contains documentation of performance on each subordinate throughout the year.','pos'),
('Has had no unrated PPR\'s or untimely planning sessions in this rating year. Always completes PPRs well within the 60 day deadline date.','pos'),
('Is proactive in performance evaluations.','pos'),
('Has an open door policy for all subordinates.','pos'),
('Encourages employees to improve knowledge, abilities and skills.','pos'),
('Maintains detailed written performance documentation that needs no explanation.','pos')
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]
dt_classifier = classifiers.DecisionTreeClassifier(training)
def clean_sentence(sentence): 
    ''' 
    Utility function to clean sentence text by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence).split()) 

def get_sentence_sentiment(sentence, classifier): 
    ''' 
    Utility function to classify sentiment of passed sentence 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed sentence text 
    analysis = TextBlob(clean_sentence(sentence), classifier=classifier) 
    print(analysis.sentiment)
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'

def get_paragraph_sentiment(paragraph):
    results = []
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        res = {}
        res['sentence'] = sentence
        res['sentiment'] = get_sentence_sentiment(sentence, dt_classifier)
        results.append(res)
    return (paragraph, results)

# if __name__ == '__main__':
#     paragraph = "Has made frequent errors that are harmful to business operations. The supervisor/department head has received numerous complaints about the quality of work. The quality of work produced is unacceptable. Does not complete required paperwork. Does not require constant supervision. Error rate is acceptable, and all work is completed timely. Forms and required paperwork are completed on time with minimal errors."
#     print(get_paragraph_sentiment(paragraph))