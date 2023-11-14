
# DeepEval - Unit Testing for LLMs
# https://docs.confident-ai.com/
# https://github.com/confident-ai/deepeval
# > deepeval test run test_by_DeepEval.py


from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.conceptual_similarity import ConceptualSimilarityMetric

from deepeval.run_test import run_test
from deepeval.test_case import LLMTestCase


def my_run_test(input, actual_output, expected_output, context):
    ## Factual Consistency: [actual_output, context]
    # Test for factual consistency
    # actual_output - the LLM generated text
    # context - the text from which the LLM is supposed to reason and derive conclusions from
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )
    metric = FactualConsistencyMetric(minimum_score=0.8)
    run_test(test_case, metrics=[metric])

    ## Answer Relevancy: [input, actual_output]
    # Ensure the answer is relevant
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )
    metric = AnswerRelevancyMetric(minimum_score=0.8)
    run_test(test_case, metrics=[metric])

    ## Conceptual Similarity: [actual_output, expected_output]
    # Know when queries are conceptually similar
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )
    metric = ConceptualSimilarityMetric(minimum_score=0.6)
    run_test(test_case, metrics=[metric])


test_list = [

    # Basic information --------------------------------------------------------------------
    #1
    {'input':
         '''Who takes headmaster?''',

     'actual_output':
         '''The headmaster of Radley College is Mr. John Moule MA.''',

     'expected_output':
         '''The headmaster of Radley College is Mr John Moule.''',

     'context':
         '''John Stuart Moule is the Warden of Radley College, and a former Head Master of Bedford School.'''
     },

    #2
    {'input':
         '''What is the colour of Social A?''',

     'actual_output':
         '''The colour of Social A at Radley College is blue and brown.''',

     'expected_output':
         '''The Social A colours are blue and brown.''',

     'context':
         '''Social Name is A. Colours are Blue and brown.'''
     },

    # For international students -----------------------------------------------------------------
    #3
    {'input':
         '''Will there be an EAL level test for international students when they arrive at your school?''',

     'actual_output':
         '''Yes, international students are assessed upon their arrival at Radley College.''',

     'expected_output':
         '''As soon as the students arrive at Radley College, they takes assessment.''',

     'context':
         '''In Radley College, it is expected that EAL students who may have EAL needs and require support upon arrival are identified.'''
     },

    #4
    {'input':
         '''What is needed for EAL students to apply for university about English language proficiency?''',

     'actual_output':
         '''Universities may accept IELTS (International English Language Testing System), TOEFL (Test of English as a Foreign Language), Cambridge English Exams (e.g., Cambridge English: Advanced or Proficiency) or Pearson Test of English (PTE) Academic.''',

     'expected_output':
         '''IELTS, TOEFL''',

     'context':
         '''English competency for university preparation – i.e. IELTS'''
     },

    # Radley Open Day (11st November 2023) ---------------------------------------------------------------------
    #5
    {'input':
         '''How many pupils in school?''',

     'actual_output':
         '''There were 686 pupils in Radley College.''',

     'expected_output':
         '''Number of pupils: 687''',

     'context':
         '''Radley College is an independent boarding school of about 680 boys.'''
     },

    #6
    {'input':
         '''University destination?''',

     'actual_output':
         '''UK Destinations:
UWE Bristol
Cardiff Metropolitan University
University of Essex
University of Kent
Lancaster University
Loughborough University
University of Nottingham
Newcastle University
Oxford Brookes University
University of Plymouth
Royal Agriculture University
University of Birmingham
University of Southampton
University of Strathclyde
University of Manchester
Durham University
University of Exeter
University of Bristol
Oxford University
University of Warwick
University of Edinburgh

USA Destinations:
University of North Carolina
Harvard College
Pomona College
University of Virginia
New York University (NYU)

Other Destinations:
Vrije Universiteit, Amsterdam
University of Bocconi, Milan
University of the Arts London
University College London
Imperial College London
LSE (London School of Economics)
London Film Academy''',

     'expected_output':
         '''University of Edinburgh, Durham University, University of Manchester, University of Warwick, Oxford University, University of Bristol and more 15 UK universities, USA destinations and others''',

     'context':
         '''2022 LEAVERS’ DESTINATIONS
University of Edinburgh (5)
Durham University (8) University of Manchester (1)
University of Warwick (3) Oxford University (9) University of Bristol (9)
University of Exeter (7)
Other UK Destinations
UWE Bristol (1)
Cardiff Metropolitan University (1) University of Essex (1)
University of Kent (1)
Lancaster University (1) Loughborough University (2) University of Nottingham (1) Newcastle University (14)
Oxford Brookes University (6) University of Plymouth (1)
Royal Agriculture University (1) University of Birmingham (1) University of Southampton (2) University of Strathclyde (1)

USA Destinations
University of North Carolina (1) Harvard College (1)
Pomona College (1) University of Virginia (1)
New York University (NYU) (1)

Other Destinations
Vrije Universiteit, Amsterdam (1) University of Bocconi, Milan (1)
University of the Arts London (1) University College London (4) Imperial College London (3)
LSE (1)
London Film Academy (1)'''
     },


]


for idx, set in enumerate(test_list):
    print(idx)
    my_run_test(set['input'], set['actual_output'], set['expected_output'], set['context'])
