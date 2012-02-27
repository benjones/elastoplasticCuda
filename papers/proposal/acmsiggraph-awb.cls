%
% ``acmsiggraph'' LaTeX class.
% - borrows *heavily* from Daniel Kartch's 'siggraph' class. 
% - also uses pieces of 'apalike.sty' and 'authordate1-4.cls'
% - modifications include adjustment of citation format to more closely
%   align with new (for SIGGRAPH 2002 conference and beyond) reference
%   and citation formatting requirements.
% - version 0.1 2001/06/01 Stephen Spencer (spencer@acm.org)
% - version 0.2 2001/10/15 Stephen Spencer
%   - the "\onlineid" variable works with the "review" mode, placing a
%     banner across the top of each page "Online Submission ID 'onlineid'
% - version 0.3 2002/01/11 Stephen Spencer
%   - disabled the cover page option.
% - version 0.4 2002/01/23 Stephen Spencer
%   - based on suggestions from James O'Brien, the following changes:
%     - correction if '\ifcamera' and '\ifcameraelse' commands
%     - page numbering in both review and preprint modes
%     have been made.
% - version 0.5 2005/01/22 Stephen Spencer
%   - removed the category markers
% - version 0.6 2006/12/15 Stephen Spencer
%   - tightened up space around section headings
% - version 0.7 2008/10/21 Stephen Spencer
%   - added 'annualconference' parameter for copyright space
%   - added line number support for 'review' mode
%

% Several changes have been made to this file by AWB.  They
% are marked in the file.  They mimic changes by JOB.

% ------------ identification --------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{acmsiggraph-awb}[2008/10/21 ACM SIGGRAPH]
% -- Begin change by AWB
% The cite package makes working with the new citation format a little
% less objectionable.  But one cannot include cite after acmsiggraph
% has already redefined the \cite command.  So it needs to be loaded
% first.  If there were a reason why someone might NOT want to load
% the cite package, then it should be made a class option to
% acmsiggraph.
\RequirePackage{cite}
% -- End change by AWB

% ------------ initial code --------------

\newif\ifacm@camera
\newif\ifacm@preprint
% -- Begin change by AWB
% Prep-rints are nice, but many people like to post a copy of
% the paper after it has been accepted, a re-print.  Lines marked
% with ADDED by AWB (reprint mod)
\newif\ifacm@reprint
% -- End change by AWB
\newif\ifacm@review
\newif\ifacm@doublespaced
% -- Begin change by JOB
% The old version of the SIGGRAPH proceedings has some nice formatting
% with a headrule and  headers with the name of the conference.  This
% option adds those headers.  In addition to specifying reprintyear, and
% reprintlocation, you also give reprintdates and optionally reprintlogo.
\newif\ifacm@pretty 
\acm@prettyfalse
% -- End change by JOB
\newif\ifacm@annualconference


\newcommand{\acm@columnmode}{}
\newcommand{\acm@pointsize}{}
\newcommand{\acm@copyrightspaceheight}{1.0in}

% ------------ required package(s) --------------

\RequirePackage{lineno}

% ------------ declaration of options --------------

\DeclareOption{pretty}{%
  \acm@prettytrue }


% 'cameraready' option.

\DeclareOption{cameraready}{%
    \acm@cameratrue%
    \acm@preprintfalse%
    \acm@reprintfalse% ADDED by AWB (reprint mod)
    \acm@reviewfalse%
    \renewcommand{\acm@columnmode}{onecolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{11pt}}

% 'review' option.

\DeclareOption{review}{%
    \acm@camerafalse%
    \acm@preprintfalse%
    \acm@reprintfalse% ADDED by AWB (reprint mod)
    \acm@reviewtrue%
    \renewcommand{\acm@columnmode}{twocolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{9pt}}

% 'widereview' option.

\DeclareOption{widereview}{%
    \acm@camerafalse%
    \acm@preprintfalse%
    \acm@reprintfalse% ADDED by AWB (reprint mod)
    \acm@reviewtrue%
    \renewcommand{\acm@columnmode}{onecolumn}%
    \acm@doublespacedtrue%
    \renewcommand{\acm@pointsize}{12pt}}

% 'preprint' option.

\DeclareOption{preprint}{%
    \acm@camerafalse%
    \acm@preprinttrue%
    \acm@reprintfalse% ADDED by AWB (reprint mod)
    \acm@reviewfalse%
    \renewcommand{\acm@columnmode}{twocolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{9pt}}


% -- Begin change by AWB
% Reprint option (reprint mod)
\DeclareOption{reprint}{%
    \acm@camerafalse%
    \acm@preprintfalse%
    \acm@reprinttrue%
    \acm@reviewfalse%
    \renewcommand{\acm@columnmode}{twocolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{9pt}}
% -- End change by AWB

% Assume, for the moment, that we're preparing a 'cameraready' version.

\ExecuteOptions{cameraready}

% Here's a warning command for use just below.

\newcommand{\acm@optwarning}[1]{%
    \ifacm@camera
        \ClassWarningNoLine{acmsiggraph}%
            {#1 option used in camera-ready mode.\MessageBreak
             This violates submission specifications}
    \fi
}

% The 'onecolumn' option doesn't work in 'cameraready' mode.

\DeclareOption{onecolumn}{%
    \acm@optwarning{onecolumn}
    \renewcommand{\acm@columnmode}{onecolumn}}

% The 'twocolumn' option works in 'cameraready' mode.

\DeclareOption{twocolumn}{%
    \renewcommand{\acm@columnmode}{twocolumn}}

% Only the '9pt' size works in 'cameraready' mode.

\DeclareOption{9pt}{%
    \renewcommand{\acm@pointsize}{9pt}}
\DeclareOption{10pt}{%
    \acm@optwarning{10pt}
    \renewcommand{\acm@pointsize}{10pt}}
\DeclareOption{11pt}{%
    \acm@optwarning{11pt}
    \renewcommand{\acm@pointsize}{11pt}}
\DeclareOption{12pt}{%
    \acm@optwarning{12pt}
    \renewcommand{\acm@pointsize}{12pt}}

% The 'singlespace' option works in 'cameraready' mode.

\DeclareOption{singlespace}{%
    \acm@doublespacedfalse}

% The 'doublespace' option does not work in 'cameraready' mode.

\DeclareOption{doublespace}{%
    \acm@optwarning{doublespace}
    \acm@doublespacedtrue}

% No 'titlepage' option in 'cameraready' mode.

\DeclareOption{titlepage}{%
    \OptionNotUsed%
    \ClassWarningNoLine{acmsiggraph}{titlepage option not allowed}}

% No 'landscape' mode in 'cameraready' mode, either.

\DeclareOption{landscape}{%
    \OptionNotUsed%
    \ClassWarningNoLine{acmsiggraph}{landscape option not allowed}}

% The mandated copyright space is 1.5 inches for the annual conferences
% and 1.0 inches for other sponsored events.

\DeclareOption{annualconference}{%
  \renewcommand{\acm@copyrightspaceheight}{1.5in}}

% Pass everything else to the 'artlcle' class, upon which this is based.

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\PassOptionsToClass{\acm@columnmode}{article}
\ifdim\acm@pointsize>9pt
    \PassOptionsToClass{\acm@pointsize}{article}
\fi

% ------------ package loading --------------

\LoadClass{article}

% ------------ main code --------------

\newcommand{\acm@onlineid}{}
\newcommand{\onlineid}[1]{\renewcommand{\acm@onlineid}{#1}}

\newcommand{\acm@preprinttext}{Submitted to SIGGRAPH 2012 - DO NOT DISTRIBUTE}
\newcommand{\preprinttext}[1]{\renewcommand{\acm@preprinttext}{#1}}

% -- Begin change by AWB (reprint/pretty mod)
\newcommand{\acm@reprintyear}{????}
\newcommand{\reprintyear}[1]{\renewcommand{\acm@reprintyear}{#1}}

\newcommand{\acm@reprintlocation}{}
\newcommand{\reprintlocation}[1]{\renewcommand{\acm@reprintlocation}{#1}}

\newcommand{\acm@reprinttext}{From the ACM SIGGRAPH \acm@reprintyear~conference proceedings.}
\newcommand{\reprinttext}[1]{\renewcommand{\acm@reprinttext}{#1}}

\newcommand{\acm@reprintdates}{Month, ??--??}
\newcommand{\reprintdates}[1]{\renewcommand{\acm@reprintdates}{#1}}

\newif\ifacm@uselogo 
\acm@uselogofalse
\newcommand{\acm@reprintlogo}{}
\newcommand{\reprintlogo}[1]{\acm@uselogotrue\renewcommand{\acm@reprintlogo}{#1}}
\newcommand{\acm@reprintlogobefore}{}
\newcommand{\reprintlogobefore}[1]{\renewcommand{\acm@reprintlogobefore}{#1}}
% -- End change by AWB 



\newcommand{\acm@reviewtext}{Online Submission ID: \acm@onlineid}
\newcommand{\reviewtext}[1]{\renewcommand{\acm@reviewtext}{#1}}

\newcommand{\ifcamera}[1]{\ifacm@camera #1 \fi}
\newcommand{\ifreview}[1]{\ifacm@review #1 \fi}
\newcommand{\ifcameraelse}[2]{\ifacm@camera #1 \else #2 \fi}
\newcommand{\ifreviewelse}[2]{\ifacm@review #1 \else #2 \fi}

\setlength{\textheight}{9.250in}
\setlength{\textwidth}{7.000in}
\setlength{\topmargin}{-0.700in}
\setlength{\oddsidemargin}{-0.250in}
\setlength{\evensidemargin}{-0.250in}
\setlength{\columnsep}{2pc}
\setlength{\headheight}{0.200in}
\setlength{\headsep}{0.250in}
\setlength{\footskip}{0.500in}
\setlength{\parindent}{1em}
\flushbottom

%% citation and reference definitions.

% enclose citations in square brackets.

% no labels in bibliography.

\def\@biblabel#1{}

% set the first line of each reference flush left, and each succeeding
% line with a 1em indentation.

\newlength{\bibhang}
\setlength{\bibhang}{1em}

%%%\def\thebibliography#1{\par\footnotesize
\def\thebibliography#1{%
  \section*{%
    \refname\@mkboth{\sl\uppercase{\refname}}{\sl\uppercase{\refname}}}
  \list{\relax}{\setlength{\labelsep}{0em}
                \setlength{\itemindent}{-\bibhang}
                \setlength{\leftmargin}{\bibhang}}
  \def\newblock{\hskip .11em plus .33em minus .07em}
  \sloppy\clubpenalty4000\widowpenalty4000
  \sfcode`\.=1000\relax}

\def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
  \def\@citea{}\@cite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{; }\@ifundefined
      {b@\@citeb}{{\bf ?}\@warning
      {Citation '\@citeb' on page \thepage \space undefined}}%
{\csname b@\@citeb\endcsname}}}{#1}}

\let\@internalcite\cite
\def\cite{\def\citename##1{##1}\@internalcite}
\def\shortcite{\def\citename##1{}\@internalcite}

%%

\renewcommand{\ps@plain}%
    {%
    \renewcommand{\@oddhead}{}%
    \renewcommand{\@oddfoot}{}%

    \ifacm@preprint%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@preprinttext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%


% -- Begin change by AWB (reprint mod)
    \ifacm@reprint%
      \renewcommand{\@oddhead}{}
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%
% -- End change by AWB


    \ifacm@review%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@reviewtext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

    \renewcommand{\@evenhead}{\@oddhead}%
    \renewcommand{\@evenfoot}{\@oddfoot}%
    }

\renewcommand{\ps@empty}%
    {%
    \renewcommand{\@oddhead}{}%
    \renewcommand{\@oddfoot}{}%

    \ifacm@preprint%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@preprinttext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

% -- Begin change by AWB (reprint mod)
    \ifacm@reprint%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@reprinttext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%
% -- End change by AWB

    \ifacm@review%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@reviewtext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

    \renewcommand{\@evenhead}{\@oddhead}%
    \renewcommand{\@evenfoot}{\@oddfoot}%
    }

% no page numbers - they are added in production

%----------------------------------------------------------------------------
% -- Begin change by AWB
% Set up pretty mode... 
\ifacm@pretty%
  \typeout{Using pretty SIGGRAPH mode.}

  \newcommand{\acm@headI}{}
  \newcommand{\acm@headO}{%
    \ifodd\thepage%
      {Computer Graphics Proceedings, Annual Conference Series, \acm@reprintyear}%
    \else%
      {ACM SIGGRAPH \acm@reprintyear, \acm@reprintlocation, \acm@reprintdates, \acm@reprintyear}%
    \fi%
  }

  \newcommand{\acm@headF}{Computer Graphics Proceedings, Annual Conference Series, \acm@reprintyear}

  \newcommand{\acm@footI}{}
  \newcommand{\acm@footO}{\thepage}

  \newcommand{\acm@headR}{\ifodd\thepage{\acm@headO}\else{\acm@headI}\fi}
  \newcommand{\acm@headL}{\ifodd\thepage{\acm@headI}\else{\acm@headO}\fi}

  \newcommand{\acm@footR}{\ifodd\thepage{\acm@footO}\else{\acm@footI}\fi}
  \newcommand{\acm@footL}{\ifodd\thepage{\acm@footI}\else{\acm@footO}\fi}

  \renewcommand{\ps@plain}{%
    \renewcommand{\@oddhead}{%
      {\fontsize{8}{8}\textsf{%
        \makebox[0in][l]{\rule[-0.75ex]{\textwidth}{0.4pt}}\mbox{\acm@headL}\hfill\mbox{\acm@headR}}%
      }}
    \renewcommand{\@oddfoot}{%
      {\fontsize{8}{8}\textsf{%
        \mbox{\acm@footL}\hfill\mbox{\acm@footR}}%
      }}
    \renewcommand{\@evenhead}{\@oddhead}
    \renewcommand{\@evenfoot}{\@oddfoot}
  }

  \renewcommand{\ps@empty}{%
    \renewcommand{\@oddhead}{%
      {\fontsize{8}{8}\textsf{%
        \makebox[0in][l]{\rule[-0.75ex]{\textwidth}{0.4pt}}%
        \raisebox{-.33in}[0in][0in]{\makebox[0in][l]{%
           \colorbox{white}{\ifacm@uselogo\acm@reprintlogobefore\includegraphics[height=.66in]{\acm@reprintlogo}\fi~~~}%
        }}
        \hfill\mbox{\acm@headF}}%
      }}
    \renewcommand{\@oddfoot}{%
      {\fontsize{8}{8}\textsf{%
        \mbox{\acm@footL}\hfill\mbox{\acm@footR}}%
      }}
    \renewcommand{\@evenhead}{\@oddhead}
    \renewcommand{\@evenfoot}{\@oddfoot}
  }
\fi
% -- End change by AWB
%----------------------------------------------------------------------------


\pagestyle{plain}

% if we are in 'review' mode turn on line numbers.

\ifacm@review%
  \linenumbers
\fi%

\newcommand{\acm@setninepoint}{
    \renewcommand\normalsize{%
        \@setfontsize\normalsize\@ixpt\@xpt
        \abovedisplayskip 9\p@ \@plus2\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus3\p@
        \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
        \belowdisplayskip \abovedisplayskip
        \let\@listi\@listI}
    \renewcommand\small{%
        \@setfontsize\small\@viipt\@ixpt
        \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus2\p@
        \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 4\p@ \@plus2\p@ \@minus2\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip}
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize\@viiipt{9.5}%
        \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus\p@
        \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 3\p@ \@plus\p@ \@minus\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip}
    \renewcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
    \renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
    \renewcommand\large{\@setfontsize\large\@xpt\@xiipt}
    \renewcommand\Large{\@setfontsize\Large\@xiipt{14}}
    \renewcommand\LARGE{\@setfontsize\LARGE\@xivpt{18}}
    \renewcommand\huge{\@setfontsize\huge\@xviipt{22}}
    \renewcommand\Huge{\@setfontsize\Huge\@xxpt{25}}
    \selectfont
}
\ifdim\acm@pointsize=9pt
    \acm@setninepoint
\fi

\newcommand{\acm@sectionfont}{}
\newcommand{\sectionfont}[1]{\renewcommand{\acm@sectionfont}{#1}}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                {-1.0ex}%
                {1.0ex}%
                {\reset@font\Large\sffamily\bfseries\acm@sectionfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                {-1.0ex}%
                {1.0ex}%
                {\reset@font\large\sffamily\bfseries\acm@sectionfont}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                {-1.0ex}%
                {1.0ex}%
                {\reset@font\normalsize\sffamily\bfseries\acm@sectionfont}}

\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                {2.05ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\acm@sectionfont}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                {3.25ex \@plus1ex \@minus .2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\acm@sectionfont}}

\newcommand{\acm@empty}{}

\newcommand{\acm@affiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\acm@affiliation}{#1}}

\newcommand{\acm@format}{}
\newcommand{\format}[1]{\renewcommand{\acm@format}{#1}}
\newcommand{\acmformat}[1]{\format{#1}}

\newcommand{\acm@teaser}{}
\newcommand{\teaser}[1]{\renewcommand{\acm@teaser}{#1}}

\newlength{\titlespace}
\setlength{\titlespace}{0.25in}

\newlength{\teaserspace}
\setlength{\teaserspace}{0.25in}

\ifacm@review
    \renewcommand{\@maketitle}{%
        \begin{center}%
            \renewcommand{\thanks}[1]{}
            {\LARGE\sffamily\bfseries\acm@sectionfont \@title \par}%
            \vspace{1\baselineskip}%
            \vspace{\titlespace}%
            \ifx\acm@teaser\acm@empty \else%
                \begingroup%
                    \def\@captype{figure}%
                    \acm@teaser%
                \endgroup\par%
                \vspace{\teaserspace}%
            \fi%
        \end{center} \par%
    }
\else
    \renewcommand{\@maketitle}{%
        \begin{center}%
            {\LARGE\sffamily\bfseries\acm@sectionfont \@title \par}%
            \vspace{1\baselineskip}%
            \large
            \begin{tabular}[t]{c}%
                \@author
            \end{tabular}\par%
            \ifx\acm@affiliation\acm@empty \else%
                \vspace{0.5\baselineskip}%
                \acm@affiliation\par%
            \fi%
            \vspace{\titlespace}%
            \ifx\acm@teaser\acm@empty \else%
                \begingroup%
                    \def\@captype{figure}%
                    \acm@teaser%
                \endgroup\par%
                \vspace{\teaserspace}%
            \fi%
        \end{center} \par%
    }
\fi

\let\acm@origmaketitle\maketitle
\let\acm@origand\and
\renewcommand{\maketitle}{%
    \let\acm@title\@title%
    \let\acm@author\@author%
    \acm@origmaketitle%
    \thispagestyle{empty}%
    \ifacm@doublespaced%
        \renewcommand{\baselinestretch}{1.66}\selectfont%
    \fi
}

\newtoks\acm@abs
\ifacm@review
    \long\def\acm@add#1{\global\acm@abs\expandafter{\the\acm@abs#1}}
    \long\def\acm@collect{%
        \global\acm@abs{}%
        \let\abstract\acm@@collect
        \abstract
    }
    \long\def\acm@@collect#1\end#2{%
        \def\@tempa{#2}%
        \ifx\@tempa\@currenvir
            \acm@add{#1}%
            \edef\abstract{\noexpand\end{\@tempa}}%
        \else
            \acm@add{#1\end{#2}}%
        \fi
        \abstract
    }
    \newcommand{\acm@modify}{%
        \let\acm@origabs\abstract%
        \let\acm@origendabs\endabstract%
        \renewenvironment{abstract}%
            {\acm@collect}%
            {\begingroup
                \let\abstract\acm@origabs
                \let\endabstract\acm@origendabs
                \begin{abstract} \the\acm@abs \end{abstract}
            \endgroup}%
    }
    \AtBeginDocument{\acm@modify}
\fi
\newcommand{\keywordlist}{%
    \vspace{0.5\baselineskip}%
    \par\noindent \textbf{Keywords: } \acm@keywords%
}
\newenvironment{CRcatlist}{%
    \vspace{0.5\baselineskip}%
    \par\noindent \textbf{CR Categories: }%
}{}

\newcommand{\CRcat}[4]{%
    #1 [#2]%
    \ifx#3\acm@empty \else : #3\fi%
    \ifx#4\acm@empty \else ---#4\fi%
}

% leave a 1.5 inch space at the bottom of the left column
% on the first page for the copyright block.

\newlength{\acm@copyrightlength}
\setlength{\acm@copyrightlength}{1.5in}

% -- Begin change by AWB
% Many people like to put their email addresses above the copyright
% space, so the \abovecopyrightspacetext command is added.  ACM also
% wants a copyright notice to appear on submitted and preprint versions
% of the text so I added \insidecopyrightspacetext with appropriate default
% values. (Sorry about the long command names.)
\newcommand{\acm@defaultinsidecopyrightspacetext}{%
  \ifacm@camera\fi
  \ifacm@review
    \textsf{%
      This work has been submitted for publication. Copyright may be
      transferred without further notice and the accepted version may
      then be posted by the publisher.  Unauthorized viewing,
      duplication, and/or distribution are prohibited.
    }%
  \fi
  \ifacm@preprint
    % This is not the right notice... it should be changed to whatever
    % ACM wants.
    \textsf{%
      \ifacm@pretty\vfill\vfill\textbf{\acm@preprinttext}\vfill\fi
      This work has been submitted or accepted for publication. Copyright may be
      transferred without further notice and the accepted version may
      then be posted by the publisher. 
    }%
  \fi
  \ifacm@reprint
    \textsf{
      \ifacm@pretty\vfill\vfill\textbf{\acm@reprinttext}\vfill\fi
      \begin{minipage}{\columnwidth}
        \renewcommand{\baselinestretch}{1}\fontsize{7}{7}\selectfont% 
Copyright\copyright~\acm@reprintyear~by the Association for Computing Machinery,
Inc. Permission to make digital or hard copies of part of this work
for personal or classroom use is granted without fee provided that
copies are not made or distributed for profit or commercial advantage
and that copies bear this notice and the full citation on the first
page or intial screen of the document. Copyrights for components of
this work owned by others than ACM must be honored. Abstracting with
credit is permitted. To copy otherwise, to republish, to post on
servers, or to redistribute to lists, requires prior specific
permission and/or a fee. Request permissions from Publications Dept.,
ACM Inc., fax +1 (212) 869-0481, or permissions@acm.org.\\
\\
        \textbf{ACM SIGGRAPH \acm@reprintyear, \acm@reprintlocation} \\
      \end{minipage}      
    }%
  \fi
}     
\newcommand{\acm@abovecopyrightspacetext}{}
\newcommand{\abovecopyrightspacetext}[1]{\renewcommand{\acm@abovecopyrightspacetext}{\par\noindent #1\\}}
\newcommand{\acm@insidecopyrightspacetext}{\acm@defaultinsidecopyrightspacetext}
\newcommand{\insidecopyrightspacetext}[1]{\renewcommand{\acm@insidecopyrightspacetext}{#1}}
\newcommand{\copyrightspace}{%
  \renewcommand{\thefootnote}{}%
  \footnotetext[0]{%
    \par\noindent\vspace{-4ex}%
    \acm@abovecopyrightspacetext%
    \parbox[t][\acm@copyrightlength][b]{\columnwidth}{%
      \ifacm@camera
      \else
        \acm@insidecopyrightspacetext
 	\vspace{1.25ex}%
      \fi
    }%
  }%
  \renewcommand{\thefootnote}{\arabic{footnote}}%
}
% -- Previous code 
%\newcommand{\copyrightspace}{%
%    \renewcommand{\thefootnote}{}%
%    \footnotetext[0]{\rule[\acm@copyrightlength]{2.71828in}{0in}}%
%    \renewcommand{\thefootnote}{\arabic{footnote}}
%}
% -- End change by AWB

\renewcommand{\footnoterule}{%
    \kern-3pt
    \hrule width 0.45\columnwidth
    \kern 2.6pt}

\newcommand{\acm@contactname}{}
\newcommand{\contactname}[1]{\renewcommand{\acm@contactname}{#1}}

\newcommand{\acm@contactaddress}{}
\newcommand{\contactaddress}[1]{\renewcommand{\acm@contactaddress}{#1}}

\newcommand{\acm@contactphone}{}
\newcommand{\contactphone}[1]{\renewcommand{\acm@contactphone}{#1}}

\newcommand{\acm@contactfax}{}
\newcommand{\contactfax}[1]{\renewcommand{\acm@contactfax}{#1}}

\newcommand{\acm@contactemail}{}
\newcommand{\contactemail}[1]{\renewcommand{\acm@contactemail}{#1}}

\newcommand{\acm@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\acm@keywords}{#1}}

\newcommand{\acm@estpages}{}
\newcommand{\estpages}[1]{\renewcommand{\acm@estpages}{#1}}

\newif\ifacm@cover
\acm@coverfalse
%\ifacm@review
%    \acm@covertrue
%\else
%    \acm@coverfalse
%\fi

\newcommand{\suppresscover}{\acm@coverfalse}
\newcommand{\acm@coverpage}{%
    \begin{titlepage}%
        \def\thanks##1{}
        \let\and\acm@origand
        \acm@setninepoint\normalsize
        \renewcommand{\baselinestretch}{1}\selectfont
        \begin{center}%
            \vspace*{\fill}
            {\LARGE\sffamily\bfseries\acm@sectionfont \acm@title \par}%
            \vspace{2\baselineskip}%
            {\large
                \begin{tabular}[t]{c}%
                \acm@author
             \end{tabular}\par%
            }%
            \vspace{1\baselineskip}%
            {\large \acm@affiliation \par}%
            \addvspace{3\baselineskip}%
            \vspace{0.5\baselineskip}%
            {Format: \acm@format \par}%
            \vspace{3\baselineskip}%
            \begin{tabular}{ll}
                Contact: & \acm@contactname \\[1\baselineskip]
                         & \begin{tabular}[b]{@{}l@{}}
                               \acm@contactaddress
                               \end{tabular} \\[1\baselineskip]
                phone: & \acm@contactphone \\
                fax: & \acm@contactfax \\
                email: & \acm@contactemail
            \end{tabular}\par%
            \vspace*{3\baselineskip}%
            {Estimated \# of pages: \acm@estpages \par}%
            \vspace*{\baselineskip}%
            {Keywords: \acm@keywords \par}%
            \vspace*{\fill}%
            \begin{minipage}{5in}%
                \the\acm@abs
            \end{minipage}\par%
            \vspace*{\fill}
        \end{center}%
    \end{titlepage}%
}

\AtEndDocument{\ifacm@cover \acm@coverpage \fi}
\newcommand{\acmopening}[1]{%
  \ClassError{acmsiggraph}{%
    The acmopening command is no longer needed.\MessageBreak%
    Switch to the maketitle command and abstract environment}{}}
\InputIfFileExists{siggraph.cfg}
        {\typeout{***************************************^^J%
                  * Local config file siggraph.cfg used *^^J%
                  ***************************************}}
        {}
\endinput

% End of file 'acmsiggraph.cls'.

